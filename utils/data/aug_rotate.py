import torch
import torchvision.transforms.functional as F

def rotate_coil_iamge(tensor_input)
    # 4차원 텐서를 생성합니다. 예시로 크기가 (batch_size, channels, height, width)인 텐서를 가정합니다.
    # 여기서 batch_size는 이미지의 개수, channels는 이미지의 채널 수 (RGB인 경우 3)를 의미합니다.
    # 만약 이미지가 하나라면 (1, 3, height, width) 형태가 됩니다
    remainder = 20-tensor_input.shape[0]
    # 회전할 각도를 지정합니다. 양수면 시계 방향으로, 음수면 반시계 방향으로 회전합니다.
    angle = 360 // remainder
    rotated_image_tensor = tensor_input
    # 이미지의 가로(width)와 세로(height) 차원에만 회전을 적용합니다.
    for _ in range(remainder):
        rotated_image_tensor = F.rotate(rotated_image_tensor, angle, resample=False, dims=(2, 3))
        tensor_input = torch.concat((tensor_input, rotated))
        
    # 회전된 이미지 텐서를 출력합니다.
    print(rotated_image_tensor.shape)

    return rotated_image_tensor