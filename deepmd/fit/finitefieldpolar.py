import warnings
import numpy as np
from typing import Optional, Tuple, List

from deepmd.env import tf
from deepmd.common import add_data_requirement, get_activation_func, get_precision, cast_precision
from deepmd.utils.network import one_layer, one_layer_rand_seed_shift
from deepmd.utils.graph import get_fitting_net_variables_from_graph_def
from deepmd.descriptor import DescrptSeA
from deepmd.fit.fitting import Fitting

from deepmd.env import global_cvt_2_tf_float
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION

class FiniteFieldFittingSeA (Fitting) :
    """
    Fit the atomic dipole with descriptor se_a
    
    Parameters
    ----------
    descrpt : tf.Tensor
            The descrptor
    neuron : List[int]
            Number of neurons in each hidden layer of the fitting net
    resnet_dt : bool
            Time-step `dt` in the resnet construction:
            y = x + dt * \\phi (Wx + b)
    sel_type : List[int]
            The atom types selected to have an atomic dipole prediction. If is None, all atoms are selected.
    seed : int
            Random seed for initializing the network parameters.
    activation_function : str
            The activation function in the embedding net. Supported options are |ACTIVATION_FN|
    precision : str
            The precision of the embedding net parameters. Supported options are |PRECISION|
    uniform_seed
            Only for the purpose of backward compatibility, retrieves the old behavior of using the random seed
    """
    def __init__ (self, 
                  descrpt : tf.Tensor,
                  neuron : List[int] = [120,120,120], 
                  resnet_dt : bool = True,
                  sel_type : List[int] = None,
                  seed : int = None,
                  activation_function : str = 'tanh',
                  precision : str = 'default',
                  uniform_seed: bool = False
    ) -> None:
        """
        Constructor
        """
        self.ntypes = descrpt.get_ntypes()
        self.dim_descrpt = descrpt.get_dim_out()
        self.n_neuron = neuron
        self.resnet_dt = resnet_dt
        self.sel_type = sel_type
        if self.sel_type is None:
            self.sel_type = [ii for ii in range(self.ntypes)]
        self.sel_mask = np.array([ii in self.sel_type for ii in range(self.ntypes)], dtype=bool)
        self.seed = seed
        self.uniform_seed = uniform_seed
        self.seed_shift = one_layer_rand_seed_shift()
        self.fitting_activation_fn = get_activation_func(activation_function)
        self.fitting_precision = get_precision(precision)
        self.dim_rot_mat_1 = descrpt.get_dim_rot_mat_1()
        self.dim_rot_mat = self.dim_rot_mat_1 * 3
        self.useBN = False

        # read efield/dfield from `filed.npy`
        add_data_requirement(key='field', ndof=3, atomic=False, must=True, high_prec=False, default=[0, 0, 0])
        self.field_avg = None
        self.field_std = None
        self.field_inv_std = None

        self.fitting_net_variables = None
        self.mixed_prec = None

    def get_sel_type(self) -> int:
        """
        Get selected type
        """
        return self.sel_type

    def get_out_size(self) -> int:
        """
        Get the output size. Should be 3
        """
        return 3

    def _build_lower(self,
                     start_index,
                     natoms,
                     inputs,
                     field,
                     rot_mat,
                     suffix='',
                     reuse=None
                     ):
        # cut-out inputs
        inputs_i = tf.slice(inputs,
                            [0, start_index, 0],
                            [-1, natoms, -1])
        inputs_i = tf.reshape(inputs_i, [-1, self.dim_descrpt])
        rot_mat_i = tf.slice(rot_mat,
                             [0, start_index, 0],
                             [-1, natoms, -1])
        # I don't understand this..
        rot_mat_i = tf.reshape(rot_mat_i, [-1, self.dim_rot_mat_1, 3])
        layer = inputs_i
        # add e/dfield
        ext_field = tf.tile(field, [1, natoms])
        ext_field = tf.reshape(ext_field, [-1, 3])
        ext_field = tf.cast(ext_field, self.fitting_precision)
        layer = tf.concat([layer, ext_field], axis = 1)
        
        for ii in range(0, len(self.n_neuron)):
            if ii >= 1 and self.n_neuron[ii] == self.n_neuron[ii - 1]:
                layer += one_layer(layer, self.n_neuron[ii], name='layer_' + str(ii) + suffix,
                                   reuse=reuse, seed=self.seed, use_timestep=self.resnet_dt,
                                   activation_fn=self.fitting_activation_fn, precision=self.fitting_precision,
                                   uniform_seed=self.uniform_seed, initial_variables=self.fitting_net_variables,
                                   mixed_prec=self.mixed_prec)
            else:
                layer = one_layer(layer, self.n_neuron[ii], name='layer_' + str(ii) + suffix,
                                  reuse=reuse, seed=self.seed, activation_fn=self.fitting_activation_fn,
                                  precision=self.fitting_precision, uniform_seed=self.uniform_seed,
                                  initial_variables=self.fitting_net_variables, mixed_prec=self.mixed_prec)
            if (not self.uniform_seed) and (self.seed is not None): self.seed += self.seed_shift
        # (nframes x natoms) x naxis
        final_layer = one_layer(layer, self.dim_rot_mat_1, activation_fn=None,
                                name='final_layer' + suffix, reuse=reuse, seed=self.seed,
                                precision=self.fitting_precision, uniform_seed=self.uniform_seed,
                                initial_variables=self.fitting_net_variables, mixed_prec=self.mixed_prec,
                                final_layer=True)
        if (not self.uniform_seed) and (self.seed is not None): self.seed += self.seed_shift
        # (nframes x natoms) x 1 * naxis
        final_layer = tf.reshape(final_layer, [tf.shape(inputs)[0] * natoms, 1, self.dim_rot_mat_1])
        # (nframes x natoms) x 1 x 3(coord)
        final_layer = tf.matmul(final_layer, rot_mat_i)
        # nframes x natoms x 3
        final_layer = tf.reshape(final_layer, [tf.shape(inputs)[0], natoms, 3])
        return final_layer

    @cast_precision
    def build (self, 
               input_d : tf.Tensor,
               rot_mat : tf.Tensor,
               natoms : tf.Tensor,
               input_dict: Optional[dict] = None,
               reuse : bool = None,
               suffix : str = '') -> tf.Tensor:
        """
        Build the computational graph for fitting net
        
        Parameters
        ----------
        input_d
                The input descriptor
        rot_mat
                The rotation matrix from the descriptor.
        natoms
                The number of atoms. This tensor has the length of Ntypes + 2
                natoms[0]: number of local atoms
                natoms[1]: total number of atoms held by this processor
                natoms[i]: 2 <= i < Ntypes+2, number of type i atoms
        input_dict
                Additional dict for inputs.
        reuse
                The weights in the networks should be reused when get the variable.
        suffix
                Name suffix to identify this descriptor

        Returns
        -------
        dipole
                The atomic dipole.
        """
        if input_dict is None:
            input_dict = {}
        type_embedding = input_dict.get('type_embedding', None)
        atype = input_dict.get('atype', None)

        # add e/dfield
        if self.field_avg is None:
            self.field_avg = 0.
        if self.field_inv_std is None:
            self.field_inv_std = 1.
        with tf.variable_scope('fitting_attr' + suffix, reuse = reuse):
            t_dfield = tf.constant(
                value = 3,
                name  = 'dfield',
                dtype = tf.int32
                )
            t_field_avg = tf.get_variable( 
                name        = 't_field_avg',
                shape       = 3,
                dtype       = GLOBAL_TF_FLOAT_PRECISION,
                trainable   = False,
                initializer = tf.constant_initializer(self.field_avg)
                )
            t_field_istd = tf.get_variable(
                name        = 't_field_istd',
                shape       = 3,
                dtype       = GLOBAL_TF_FLOAT_PRECISION,
                trainable   = False,
                initializer = tf.constant_initializer(self.field_inv_std)
                )

        nframes = input_dict.get('nframes')
        start_index = 0
        inputs = tf.reshape(input_d, [-1, natoms[0], self.dim_descrpt])
        rot_mat = tf.reshape(rot_mat, [-1, natoms[0], self.dim_rot_mat])

        #TODO: make the rotational invariance of field.
        field = input_dict['field']
        field = tf.reshape(field, [-1, 3])
        field = (field - t_field_avg) * t_field_istd   

        if type_embedding is not None:
            nloc_mask = tf.reshape(tf.tile(tf.repeat(self.sel_mask, natoms[2:]), [nframes]), [nframes, -1])
            atype_nall = tf.reshape(atype, [-1, natoms[1]])
            # (nframes x nloc_masked)
            self.atype_nloc_masked = tf.reshape(tf.slice(atype_nall, [0, 0], [-1, natoms[0]])[nloc_mask], [-1])  ## lammps will make error
            self.nloc_masked = tf.shape(tf.reshape(self.atype_nloc_masked, [nframes, -1]))[1]
            atype_embed = tf.nn.embedding_lookup(type_embedding, self.atype_nloc_masked)
        else:
            atype_embed = None

        self.atype_embed = atype_embed

        if atype_embed is None:
            count = 0
            outs_list = []
            for type_i in range(self.ntypes):
                if type_i not in self.sel_type:
                    start_index += natoms[2+type_i]
                    continue
                final_layer = self._build_lower(
                    start_index=start_index, 
                    natoms=natoms[2+type_i],
                    inputs=inputs, 
                    field=field,
                    rot_mat=rot_mat, 
                    suffix='_type_'+str(type_i)+suffix, 
                    reuse=reuse
                    )
                start_index += natoms[2 + type_i]
                # concat the results
                outs_list.append(final_layer)
                count += 1
            outs = tf.concat(outs_list, axis = 1)
        else:
            inputs = tf.reshape(tf.reshape(inputs, [nframes, natoms[0], self.dim_descrpt])[nloc_mask],
                                [-1, self.dim_descrpt])
            rot_mat = tf.reshape(tf.reshape(rot_mat, [nframes, natoms[0], self.dim_rot_mat_1 * 3])[nloc_mask],
                                 [-1, self.dim_rot_mat_1, 3])
            atype_embed = tf.cast(atype_embed, self.fitting_precision)
            type_shape = atype_embed.get_shape().as_list()
            inputs = tf.concat([inputs, atype_embed], axis=1)
            self.dim_descrpt = self.dim_descrpt + type_shape[1]
            inputs = tf.reshape(inputs, [nframes, self.nloc_masked, self.dim_descrpt])
            rot_mat = tf.reshape(rot_mat, [nframes, self.nloc_masked, self.dim_rot_mat_1 * 3])
            final_layer = self._build_lower(
                    start_index=0, 
                    natoms=self.nloc_masked,
                    inputs=inputs, 
                    field=field,
                    rot_mat=rot_mat, 
                    suffix='_type_'+str(type_i)+suffix, 
                    reuse=reuse
                    )
            # nframes x natoms x 3
            outs = tf.reshape(final_layer, [nframes, self.nloc_masked, 3])

        tf.summary.histogram('fitting_net_output', outs)
        return tf.reshape(outs, [-1])
        # return tf.reshape(outs, [tf.shape(inputs)[0] * natoms[0] * 3 // 3])

    def init_variables(self,
                       graph: tf.Graph,
                       graph_def: tf.GraphDef,
                       suffix : str = "",
    ) -> None:
        """
        Init the fitting net variables with the given dict

        Parameters
        ----------
        graph : tf.Graph
            The input frozen model graph
        graph_def : tf.GraphDef
            The input frozen model graph_def
        suffix : str
            suffix to name scope
        """
        self.fitting_net_variables = get_fitting_net_variables_from_graph_def(graph_def, suffix=suffix)


    def enable_mixed_precision(self, mixed_prec : dict = None) -> None:
        """
        Reveive the mixed precision setting.

        Parameters
        ----------
        mixed_prec
                The mixed precision setting used in the embedding net
        """
        self.mixed_prec = mixed_prec
        self.fitting_precision = get_precision(mixed_prec['output_prec'])