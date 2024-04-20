import random

import numpy
from PIL import Image
from helper import DEFAULT_GLAUCOMATOUS_FEATURES, inference_tasks
from model1 import predict_image
from model2 import predict_labels

def run():
    _show_torch_cuda_info()

    for jpg_image_file_name, save_prediction in inference_tasks():
        # Do inference, possibly something better performant
        ...

        print(f"Running inference on {jpg_image_file_name}")

        # is_referable_glaucoma_likelihood = random.random()
        is_referable_glaucoma_likelihood = predict_image('task1.h5', jpg_image_file_name)

        is_referable_glaucoma = is_referable_glaucoma_likelihood > 0.5

        if is_referable_glaucoma:
            # features = {
            #     k: random.choice([True, False])
            #     for k, v in DEFAULT_GLAUCOMATOUS_FEATURES.items()
            # }
            labels = predict_labels('task2.h5', jpg_image_file_name)
            features = {k: labels[i] for i, (k, v) in enumerate(DEFAULT_GLAUCOMATOUS_FEATURES.items())}
        else:
            features = None
        
        print('is_referable_glaucoma:', is_referable_glaucoma)
        print('is_referable_glaucoma_likelihood:', is_referable_glaucoma_likelihood)
        print('features:', features)
        ...

        # Finally, save the answer
        save_prediction(
            is_referable_glaucoma,
            is_referable_glaucoma_likelihood,
            features,
        )
    return 0


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
