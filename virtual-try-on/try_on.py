from utils import *
from titan import *
from nova import *

def try_on_func():
    input_format = read_json('input_format.json')
    print("input format: ")
    print(input_format)
    has_generated_titan_prompt = False
    titan_prompt = ""
    while(has_generated_titan_prompt == False or len(titan_prompt) > 512):
        top = read_jpg_to_base64_str('top.jpg')
        bottom = read_jpg_to_base64_str('bottom.jpg')
        shoes = read_jpg_to_base64_str('shoes.jpg')
        accessory = read_jpg_to_base64_str('accessory.jpg')
        top_text = describe_image(top,input_format['top_description'])["output"]["message"]["content"][0]["text"]
        bottom_text = describe_image(bottom,input_format['bottom_description'])["output"]["message"]["content"][0]["text"]
        shoes_text = describe_image(shoes,input_format['shoes_description'])["output"]["message"]["content"][0]["text"]
        accessory_text = describe_image(accessory,input_format['accessory_description'])["output"]["message"]["content"][0]["text"]
        titan_prompt = "Modify the person's clothes according to this description: " + top_text + "\n\n" + bottom_text + "\n\n" +  shoes_text + "\n\n" + accessory_text
        has_generated_titan_prompt = True
        if(len(titan_prompt)> 512):
            print("Titan prompt is too long: "  + str(len(titan_prompt)) + " characters. Retrying...")
    titan_function(titan_prompt)


