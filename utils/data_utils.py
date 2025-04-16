from PIL import Image
import torchvision.transforms as T
import torch


def load_single_image(image_path, resize=True, resize_dim=(224, 224)):
    """
    Loads a single image with PIL, applies transforms, and returns the tensor.
    Also prints shape info for clarity.
    :param image_path: Path to a single .jpg file
    :return: A PyTorch tensor of shape (1, 3, 224, 224)
    """
    # Only convert to Tensor; no resizing to 224x224
    transform = T.Compose([
        T.ToTensor()  # # from [0..255] PIL image to [0..1] float tensor, shape => (C, H, W), range [0,1]
    ])

    # If resize is True, add resizing to the transform
    if resize:
        print("Resizing image to resize_dim[0] x resize_dim[1] ...")
        # Basic transform: resize to resize_dim[0] x resize_dim[1], then convert to tensor
        transform = T.Compose([
            T.Resize(resize_dim),  # (H, W)
            T.ToTensor()  # from [0..255] PIL image to [0..1] float tensor, shape (C,H,W)
        ])

    # Load the image
    pil_img = Image.open(image_path).convert('RGB')
    print(f"Original image size: {pil_img.size} (Width x Height)")

    # Apply transforms
    img_tensor = transform(pil_img)  # shape: (3, H, W)
    print(f"Transformed image shape: {img_tensor.shape} (C, H, W)")

    # Add a batch dimension => (1, 3, H, W) i.e. (batch_size, C, H, W)
    img_tensor = img_tensor.unsqueeze(0)
    print(f"Final input shape to the model: {img_tensor.shape} (Batch, C, H, W)\n")

    return img_tensor


def test_model_with_image(model, image_tensor):
    """
    Passes a single image tensor through the model and prints output shape.
    :param model: Instance of MyCNN
    :param image_tensor: shape (1, 3, H, W)
    """
    with torch.no_grad():
        output = model(image_tensor)
    print(f"Model output shape: {output.shape} (Batch, 10)")
    print("Raw output logits:", output)
