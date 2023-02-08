import jiwer
from jiwer import wer

ground_truth = "ifølge ansa var politiet bekymret for et par toppnivåtreff som de fryktet kunne utløse en ordentlig krig om arveretten"
hypothesis   = "vi farge ansatt av politiet bekymrer et par tattende må av press som de frykter kunnet utløse en ordentlig tilgjengelig om arveretten"
transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveMultipleSpaces(),
    jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
])
error = wer(ground_truth, hypothesis, truth_transform=transformation, hypothesis_transform=transformation)
print(error)
