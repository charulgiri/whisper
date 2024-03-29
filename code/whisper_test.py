import librosa
from transformers import WEIGHTS_NAME, CONFIG_NAME, AutoModelForSpeechSeq2Seq
from datasets import load_dataset, DatasetDict
import torch
import evaluate
from typing import Any, Dict, List, Union
from transformers import WhisperModel, WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoProcessor, AutoFeatureExtractor, AutoModelForSpeechSeq2Seq
from datasets import load_dataset, DatasetDict, Dataset
from dataclasses import dataclass


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

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["normsentence_text"]).input_ids
    return batch
# example 
f = ["/home/coder/whisper/Data/k_b_11354048829037172906.wav", "/home/coder/whisper/Data/m_b_75020282905139354.wav", "/home/coder/whisper/Data/m_d_1635815542987463048.wav"]
labels = ["ifølge ansa var politiet bekymret for et par toppnivåtreff som de fryktet kunne utløse en ordentlig krig om arveretten",
          "alle kan bli opplyste det eneste i veien for å få slik innsikt er vår egen anstrengelse og negativitet",
          "muligheten for at man skulle bli tvunget til å gjennomføre spillet bak lukkede dører uten lagets støttespillere ble reist som følge av frykt for mangel på transport"]
# data_path = "/home/coder/whisper/Data/k_b_11354048829037172906.wav"
di = 0
data_path = f[di]
y, r = librosa.load(data_path, sr=16000)
print(r, len(y)) 
output_dir = "/home/coder/mymodel"
model_dir = "model_tiny"
feature_dir = "whisper_feature_extractor"
processor_dir = "whisper_processor"
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny", language="Norwegian", task="transcribe")
feature_extractor = AutoFeatureExtractor.from_pretrained(feature_dir)
processor = AutoProcessor.from_pretrained(processor_dir)
# processor = WhisperProcessor.from_pretrained("./example_pretrained_processor")

model = AutoModelForSpeechSeq2Seq.from_pretrained(model_dir)


# ds = DatasetDict()

def get_input_features(y):
    inputs = feature_extractor(y, return_tensors="pt")
    input_features = inputs.input_features[0]
    return input_features

def gen(y, di):
    audio ={}
    audio["array"] = y
    audio["sampling_rate"] = 16000
    batch = {}
    batch["normsentence_text"] = labels[di]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["normsentence_text"]).input_ids
    
    yield {"input_features":batch["input_features"], "labels": batch["labels"]}

batch = Dataset.from_generator(gen, gen_kwargs={"y":y, "di":di})
input_features =  get_input_features(y)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
metric = evaluate.load("wer")
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-tiny_nob_test",  # change to a repo name of your choice
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=3e-06,
    warmup_steps=1000,
    max_steps=4000,
    gradient_checkpointing=False,
    # fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=32,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=5,
    logging_steps=5,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    # train_dataset=nb_16k["test"],
    eval_dataset=batch,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)
eval =  trainer.evaluate()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_features = input_features.to(device)
# exit(0)
generated_ids  = model.generate(inputs=input_features)
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(eval)
print(transcription)