from deepmd.infer.deep_tensor import DeepTensor
from deepmd.common import make_default_mesh
import numpy as np
from typing import List, Optional, TYPE_CHECKING

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class DeepDipole(DeepTensor):
    """Constructor.

    Parameters
    ----------
    model_file : Path
        The name of the frozen model file.
    load_prefix: str
        The prefix in the load computational graph
    default_tf_graph : bool
        If uses the default tf graph, otherwise build a new tf graph for evaluation

    Warnings
    --------
    For developers: `DeepTensor` initializer must be called at the end after
    `self.tensors` are modified because it uses the data in `self.tensors` dict.
    Do not chanage the order!
    """

    def __init__(
        self, model_file: "Path", load_prefix: str = "load", default_tf_graph: bool = False
    ) -> None:

        # use this in favor of dict update to move attribute from class to
        # instance namespace
        self.tensors = dict(
            {
                # output tensor
                "t_tensor": "o_dipole:0",
            },
            **self.tensors
        )

        DeepTensor.__init__(
            self,
            model_file,
            load_prefix=load_prefix,
            default_tf_graph=default_tf_graph,
        )

    def get_dim_fparam(self) -> int:
        """Unsupported in this model."""
        raise NotImplementedError("This model type does not support this attribute")

    def get_dim_aparam(self) -> int:
        """Unsupported in this model."""
        raise NotImplementedError("This model type does not support this attribute")
    

class DeepFiniteFieldDipole(DeepTensor):
    """Constructor.

    Parameters
    ----------
    model_file : Path
        The name of the frozen model file.
    load_prefix: str
        The prefix in the load computational graph
    default_tf_graph : bool
        If uses the default tf graph, otherwise build a new tf graph for evaluation

    Warnings
    --------
    For developers: `DeepTensor` initializer must be called at the end after
    `self.tensors` are modified because it uses the data in `self.tensors` dict.
    Do not chanage the order!
    """

    def __init__(
        self, model_file: "Path", load_prefix: str = "load", default_tf_graph: bool = False
    ) -> None:

        # use this in favor of dict update to move attribute from class to
        # instance namespace
        self.tensors = dict(
            {
                # extra input tensor field
                "t_field": "t_field:0",
                # output tensor
                "t_tensor": "o_finitefielddipole:0",
            },
            **self.tensors
        )

        DeepTensor.__init__(
            self,
            model_file,
            load_prefix=load_prefix,
            default_tf_graph=default_tf_graph,
        )

    def get_dim_fparam(self) -> int:
        """Unsupported in this model."""
        raise NotImplementedError("This model type does not support this attribute")

    def get_dim_aparam(self) -> int:
        """Unsupported in this model."""
        raise NotImplementedError("This model type does not support this attribute")

    def eval(
            self,
            coords: np.ndarray,
            cells: np.ndarray,
            atom_types: List[int],
            field: np.ndarray,
            atomic: bool = True,
            fparam: Optional[np.ndarray] = None,
            aparam: Optional[np.ndarray] = None,
            efield: Optional[np.ndarray] = None,
            mixed_type: bool = False,
        ) -> np.ndarray:
            """Evaluate the model.

            Parameters
            ----------
            coords
                The coordinates of atoms. 
                The array should be of size nframes x natoms x 3
            cells
                The cell of the region. 
                If None then non-PBC is assumed, otherwise using PBC. 
                The array should be of size nframes x 9
            atom_types
                The atom types
                The list should contain natoms ints
            atomic
                If True (default), return the atomic tensor
                Otherwise return the global tensor
            fparam
                Not used in this model
            aparam
                Not used in this model
            efield
                Not used in this model
            mixed_type
                Whether to perform the mixed_type mode.
                If True, the input data has the mixed_type format (see doc/model/train_se_atten.md),
                in which frames in a system may have different natoms_vec(s), with the same nloc.

            Returns
            -------
            tensor
                    The returned tensor
                    If atomic == False then of size nframes x output_dim
                    else of size nframes x natoms x output_dim
            """
            # standarize the shape of inputs
            if mixed_type:
                natoms = atom_types[0].size
                atom_types = np.array(atom_types, dtype=int).reshape([-1, natoms])
            else:
                atom_types = np.array(atom_types, dtype = int).reshape([-1])
                natoms = atom_types.size
            coords = np.reshape(np.array(coords), [-1, natoms * 3])
            nframes = coords.shape[0]
            if cells is None:
                pbc = False
                cells = np.tile(np.eye(3), [nframes, 1]).reshape([nframes, 9])
            else:
                pbc = True
                cells = np.array(cells).reshape([nframes, 9])

            # sort inputs
            coords, atom_types, imap, sel_at, sel_imap = \
                self.sort_input(coords, atom_types, sel_atoms=self.get_sel_type(), mixed_type=mixed_type)

            # make natoms_vec and default_mesh
            natoms_vec = self.make_natoms_vec(atom_types, mixed_type=mixed_type)
            assert(natoms_vec[0] == natoms)

            # evaluate
            feed_dict_test = {}
            feed_dict_test[self.t_natoms] = natoms_vec
            if mixed_type:
                feed_dict_test[self.t_type] = atom_types.reshape([-1])
            else:
                feed_dict_test[self.t_type] = np.tile(atom_types, [nframes, 1]).reshape([-1])

            feed_dict_test[self.t_coord] = np.reshape(coords, [-1])
            feed_dict_test[self.t_box  ] = np.reshape(cells , [-1])
            feed_dict_test[self.t_field] = np.reshape(field  , [-1])
            if pbc:
                feed_dict_test[self.t_mesh ] = make_default_mesh(cells)
            else:
                feed_dict_test[self.t_mesh ] = np.array([], dtype = np.int32)

            if atomic:
                assert "global" not in self.model_type, \
                    f"cannot do atomic evaluation with model type {self.model_type}"
                t_out = [self.t_tensor]
            else:
                assert self._support_gfv or "global" in self.model_type, \
                    f"do not support global tensor evaluation with old {self.model_type} model"
                t_out = [self.t_global_tensor if self._support_gfv else self.t_tensor]
            v_out = self.sess.run (t_out, feed_dict = feed_dict_test)
            tensor = v_out[0]

            # reverse map of the outputs
            if atomic:
                tensor = np.array(tensor)
                tensor = self.reverse_map(np.reshape(tensor, [nframes,-1,self.output_dim]), sel_imap)
                tensor = np.reshape(tensor, [nframes, len(sel_at), self.output_dim])
            else:
                tensor = np.reshape(tensor, [nframes, self.output_dim])
            
            return tensor