from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerNoMirror200ep(nnUNetTrainer):
    """
    Custom nnU-Net v2 trainer used in this project.

    Modifications to the standard nnUNetTrainer:
      - disables mirroring augmentation;
      - sets the training duration to 200 epochs.
    """

    def initialize(self):
        super().initialize()
        self.num_epochs = 200

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        mirror_axes = None
        self.inference_allowed_mirroring_axes = None

        self.print_to_log_file(
            "CUSTOM TRAINER ACTIVE: mirroring disabled; num_epochs=200"
        )

        return (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        )
