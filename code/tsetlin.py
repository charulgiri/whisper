# from huggingface_hub import notebook_login
# notebook_login()
from datasets import load_dataset
from datasets import DatasetDict
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperFeatureExtractor
import torch
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from numpy.fft import fft, ifft
import cv2
import matplotlib
# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import numpy as np
import sys
if sys.version_info[0] == 3:
    import tkinter as tk
else:
    import Tkinter as tk
print(matplotlib.get_backend())
matplotlib.use("Agg")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["normsentence_text"]).input_ids
    return batch


nb_16k = DatasetDict()
nb_16k["train"] = load_dataset("NbAiLab/NPSC", '16K_mp3_bokmaal', split="train+validation")
nb_16k["test"]  = load_dataset("NbAiLab/NPSC", '16K_mp3_bokmaal', split="test")
nb_16k = nb_16k.remove_columns(['sentence_id', 'meeting_date', 'sentence_order', 'speaker_id', 'speaker_name', 'sentence_text', 'sentence_language_code', 'text', 'start_time', 'end_time', 'transsentence_text', 'translated',])


# feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v2")
# tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny", language="Norwegian", task="transcribe")
# processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", language="Norwegian", task="transcribe")
# nb_16k = nb_16k.map(prepare_dataset, remove_columns=nb_16k.column_names["train"], num_proc=4)
# data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# print(nb_16k["train"][0])

x = nb_16k["train"][0]["audio"]["array"][:160]
# print(x)
# print(len(x))
X = fft(x)
sr = len(X)
N = len(X)
n = np.arange(N)
T = N/sr
freq = n/T
# sampling interval
ts = 1.0/sr
t = np.arange(0,1,ts)
# t = np.arange(len(x))
plt.figure(figsize = (8, 6))
plt.stem(freq, np.abs(X), 'r')
plt.ylabel('Amplitude')
plt.savefig("audiosample_fft.png")
plt.show()
cv2.waitKey(0)
# exit(0)
# sr = 16000
# ts = 1.0/sr
# t = np.arange(0,1,ts)
# X = fft(x)
# N = len(X)
# n = np.arange(N)
# T = N/sr
# freq = n/T 

# plt.figure(figsize = (12, 6))
# plt.subplot(121)

# plt.stem(freq, np.abs(X), 'b', \
#          markerfmt=" ", basefmt="-b")
# plt.xlabel('Freq (Hz)')
# plt.ylabel('FFT Amplitude |X(freq)|')
# plt.xlim(0, 10)

# plt.subplot(122)
# plt.plot(t, ifft(X), 'r')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.tight_layout()
# plt.show()