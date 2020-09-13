# Training

  1. Download the pre-trained weights of [vgg16](https://drive.google.com/file/d/1sNrwJJxCTIJ1G_7kgXkITCXZvmrJvSe5/view?usp=sharing) and [vgg16_places365_weights](https://drive.google.com/file/d/1_onEcNKpMR1R-AzWRrY9FQtHf7NGKTX5/view?usp=sharing)
  2. Download the training data of [HDR-Synth and HDR-Real](https://drive.google.com/file/d/1muy49Pd0c7ZkxyxoxV7vIRvPv6kJdPR2/view?usp=sharing)

## Training of the Dequantization-Net
  ```
  python train_dequantization_net.py --logdir_path [output_deq_ckpt_path] --hdr_prefix [hdr_synth_training_data_path]
  ```
  
## Training of the Linearization-Net
  ```
  python train_linearization_net.py --logdir_path [output_lin_ckpt_path] --hdr_prefix [hdr_synth_training_data_path]
  ```
  
## Training of the Hallucination-Net
  ```
  python train_hallucination_net.py --logdir_path [output_hal_ckpt_path] --hdr_prefix [hdr_synth_training_data_path]
  ```
  
## Joint training of the entire pipeline and Refinement-Net
  1. Convert the real HDR-jpg paired data into tfrecords for training.
  ```
  python convert_to_tf_record.py
  ```
  2. Joint training of the entire pipeline and Refinement-Net
  ```
  python finetune_real_dataset.py --logdir_path [output_hal_ckpt_path] --tfrecords_path [converted_tfrecords_path] --deq_ckpt [pretrained_deq_ckpt] --lin_ckpt [pretrained_lin_ckpt] --deq_ckpt [pretrained_hal_ckpt]
  ```
