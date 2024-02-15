from PIL import Image

def crop_and_resize(input_image, target_resolution):
    
    # Open the input image
    # input_image = Image.open(input_image)
    input_image = input_image['image']

    # Get the size of the input image
    input_width, input_height = input_image.size

    # Get the target resolution
    target_width, target_height = target_resolution

    # Calculate the aspect ratio
    input_aspect_ratio = input_width / input_height
    target_aspect_ratio = target_width / target_height

    # Determine whether to resize or pad
    if input_width < target_width or input_height < target_height:
        # Calculate the new size while preserving the aspect ratio
        if input_aspect_ratio > target_aspect_ratio:
            new_width = target_width
            new_height = int(target_width / input_aspect_ratio)
        else:
            new_width = int(target_height * input_aspect_ratio)
            new_height = target_height

        # Create a new blank image with the target resolution
        resized_image = Image.new("RGB", (target_width, target_height))

        # Calculate the position to paste the resized image
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2

        # Paste the resized image onto the new image
        resized_image.paste(input_image.resize((new_width, new_height), Image.ANTIALIAS), (paste_x, paste_y))
    else:
        # Calculate the new size while preserving the aspect ratio
        if input_aspect_ratio > target_aspect_ratio:
            new_width = target_width
            new_height = int(target_width / input_aspect_ratio)
        else:
            new_width = int(target_height * input_aspect_ratio)
            new_height = target_height

        # Resize the image
        resized_image = input_image.resize((new_width, new_height), Image.ANTIALIAS)

    # Calculate the cropping box
    left = (new_width - target_width) / 2
    top = (new_height - target_height) / 2
    right = (new_width + target_width) / 2
    bottom = (new_height + target_height) / 2

    # Crop the resized image
    cropped_image = resized_image.crop((left, top, right, bottom))

    # Save the cropped and resized image
    # cropped_image.save(output_image_path)
    return cropped_image

# Example usage
# input_image_path = "input_image.jpg"
# output_image_path = "output_image.jpg"
# target_resolution = (800, 600)

# crop_and_resize(input_image_path, output_image_path, target_resolution)
