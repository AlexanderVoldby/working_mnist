if __name__ == '__main__':
    import torch
    import os
    from torchvision import transforms

    def normalize(tensor):
        mean = tensor.mean()
        std = tensor.std()

        normalized_tensor = (tensor - mean) / std
        return normalized_tensor

    # Folder path containing .pt files
    folder_path = "data/raw"

    # Get a list of .pt files in the folder
    file_list = [f for f in os.listdir(folder_path) if f.endswith(".pt")]

    # List to store individual tensors
    train_images = []
    train_labels = []
    test_images = None
    test_labels = None

    # Load .pt files
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        tensor = torch.load(file_path)
        if file_name.startswith("train_images"):
            train_images.append(tensor)
        elif file_name.startswith("train_target"):
            train_labels.append(tensor)
        elif file_name.startswith("test_images"):
            test_images = tensor
        else:
            test_labels = tensor

    # Stack the tensors into a single tensor
    train_images_tensor = torch.cat([t.unsqueeze(0) for t in train_images], dim=0).view(-1, 28, 28)
    train_labels_tensor = torch.cat([t.unsqueeze(0) for t in train_labels], dim=0).view(-1)
    print(train_images_tensor.shape)
    print(train_labels_tensor.shape)
    # Define normalization transform
    normalized_train_img = normalize(train_images_tensor)
    normalized_test_img = normalize(test_images)

    # Save the normalized tensor
    output_path = "data/processed/"
    torch.save(train_images_tensor, output_path + "train_img.pt")
    torch.save(train_labels_tensor, output_path + "train_target.pt")
    torch.save(test_images, output_path + "test_img.pt")
    torch.save(test_labels, output_path + "test_target.pt")