# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    Optional,
)

import torch

from deepmd.pt.model.atomic_model import (
    DPZBLLinearAtomicModel,
)

from .make_model import (
    make_model,
)

DPZBLModel_ = make_model(DPZBLLinearAtomicModel)


class DPZBLModel(DPZBLModel_):
    model_type = "ener"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        coord,
        atype,
        box: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ) -> Dict[str, torch.Tensor]:
        model_ret = self.forward_common(
            coord,
            atype,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )

        model_predict = {}
        model_predict["atom_energy"] = model_ret["energy"]
        model_predict["energy"] = model_ret["energy_redu"]
        if self.do_grad("energy"):
            model_predict["force"] = model_ret["energy_derv_r"].squeeze(-2)
            if do_atomic_virial:
                model_predict["atom_virial"] = model_ret["energy_derv_c"].squeeze(-3)
            model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
        else:
            model_predict["force"] = model_ret["dforce"]
        return model_predict

    def forward_lower(
        self,
        extended_coord,
        extended_atype,
        nlist,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ):
        model_ret = self.forward_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )

        model_predict = {}
        model_predict["atom_energy"] = model_ret["energy"]
        model_predict["energy"] = model_ret["energy_redu"]
        if self.do_grad("energy"):
            model_predict["extended_force"] = model_ret["energy_derv_r"].squeeze(-2)
            model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
            if do_atomic_virial:
                model_predict["extended_virial"] = model_ret["energy_derv_c"].squeeze(
                    -2
                )
        else:
            assert model_ret["dforce"] is not None
            model_predict["dforce"] = model_ret["dforce"]
        model_predict = model_ret
        return model_predict