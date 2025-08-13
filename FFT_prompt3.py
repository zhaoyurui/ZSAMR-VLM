import os
import re
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import time
import pickle
import logging

logging.basicConfig(level=logging.INFO)

timeGet = time.localtime()
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

processor = Blip2Processor.from_pretrained("./blip2-opt-6.7b")
model = Blip2ForConditionalGeneration.from_pretrained("./blip2-opt-6.7b",torch_dtype=torch.float16).to(device)

Modality = ['FFT2']  
ModuClass = ['16QAM', '2ASK', '2FSK', '32QAM', '64QAM', '8PSK', 'BPSK', 'OQPSK', 'QPSK', 'pi4QPSK']
ModuClass_num = len(ModuClass)
SampleNum = 100

prompt = "Question: How many blue lines are in the picture? Answer:"  
promptId = '02'

dataset_paths = "Image-Sim-v2"

for dataset_path in dataset_paths:
    dataset_name = os.path.basename(dataset_path)
    save_folder = os.path.join("./GPU_GenCaption", dataset_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for ModalIndex in range(len(Modality)): 
        ModalTemp = Modality[ModalIndex]
        logging.info("======Modality=====" + ModalTemp + "========")

        full_save_folder = os.path.join(save_folder, ModalTemp)
        if not os.path.exists(full_save_folder):
            os.makedirs(full_save_folder)
        saveFilePath = os.path.join(full_save_folder,
                                    "Test_03_Prompt_" + promptId + "_Model_" + 'BLIP2-opt-6.7b' + "_MaxLen_" + str(
                                        20) + '_Caption_Best_Sample_' + str(
                                        SampleNum) + '_Modality_' + ModalTemp + "_Date_" + \
                                    str(timeGet.tm_mon) + "_" + str(timeGet.tm_mday) + "_" + str(
                                        timeGet.tm_hour) + "_" + str(timeGet.tm_min))
        txtFilePath = saveFilePath + ".txt"
        pklFilePath = saveFilePath + ".pkl"

        result = {}
        fileTemp = open(txtFilePath, "w", buffering=1)

        for ModuIndex in range(len(ModuClass)):
            ModuTemp = ModuClass[ModuIndex]
            fileTemp.write("=====" + ModalTemp + "=====" + ModuTemp + "=====")
            fileTemp.write('\n')
            for SampleIndex in range(SampleNum):
                ImagePath = os.path.join(dataset_path, ModalTemp, ModuTemp,
                                         "Modal_" + ModalTemp + "_Modu_" + ModalTemp + "_Index_" + str(
                                             SampleIndex + 1) + ".png")
                image = Image.open(ImagePath)
                inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, dtype=torch.float16)

                generated_ids = model.generate(**inputs, max_new_tokens=20)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

                answer_start = generated_text.find("Answer:")
                if answer_start != -1:
                    answer = generated_text[answer_start + len("Answer:"):].strip()
                else:
                    answer = generated_text

                result[ModalTemp + "_" + ModuTemp + "_" + str(SampleIndex)] = answer

                fileTemp.write('Index_' + str(SampleIndex) + ":" + answer)
                fileTemp.write('\n')

        fileTemp.close()
        with open(pklFilePath, 'wb') as filePKL:
            pickle.dump(result, filePKL)
            