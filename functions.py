import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import rcParams as rc
from typing import List
import spacy
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

rc['figure.figsize'] = 12, 8
rc['font.size'] = 14


class PrimaryTextPreprocess:

    @staticmethod
    def primary_cleaning(s):
        s = str(s)
        s = s.replace('Subject:', ' ') \
            .replace('re :', ' ') \
            .replace('fw :', ' ')
        s = s.lower()
        s = re.sub(r'\W', ' ', s)
        s = s.split()
        s = " ".join(s)
        return s

    @staticmethod
    def lemmatize_and_stopwords_removal(data: pd.Series, text_col: str):
        data[text_col] = data[text_col].apply(word_tokenize)
        data[text_col] = data[text_col].apply(
            lambda x: [word for word in x if word not in set(stopwords.words('english'))])
        data[text_col] = data[text_col].apply(lambda x: ' '.join(x))


class SpacyPreprocess:

    def __init__(self, model):
        self.nlp = model

    def add_custom_pattern(self, pipe_name: str, pattern: List[dict]):
        ruler = self.nlp.add_pipe(pipe_name, before='ner')
        ruler.add_patterns(pattern)
        print(self.nlp.pipe_names)

    def concat_words(self, text):
        doc_ents = self.nlp(text).ents

        tagged_text = text
        for tag in doc_ents:
            tagged_text = re.sub(tag.text, "_".join(tag.text.split()), tagged_text)

        return tagged_text


class TrainTestPreprocess:

    def __init__(self, data):
        self.data = data

    def train_valid_test_split(self, test_size=0.2, stratify_col=None):
        train_valid, test = train_test_split(self.data, test_size=test_size, stratify=stratify_col)
        train, valid = train_test_split(train_valid, test_size=test_size, stratify=stratify_col)
        return train, valid, test

    def train_test_labels_dist(self, target: str):
        fig, ax = plt.subplots(1, 3, figsize=(20, 8))
        sns.set(font_scale=1.3)
        palette_green = sns.color_palette('Greens_d')
        palette_sky = sns.color_palette('Blues_d')
        train, valid, test = self.train_valid_test_split(stratify_col=target)
        sns.countplot(x=target, data=train, ax=ax[0], palette=palette_green)
        sns.countplot(x=target, data=valid, ax=ax[1], palette=palette_green)
        sns.countplot(x=target, data=test, ax=ax[2], palette=palette_sky)
        ax[0].set_title('Train Data')
        ax[0].set_xticks([0, 1], ['Ham', 'Spam'])
        ax[1].set_title('Valid Data')
        ax[1].set_xticks([0, 1], ['Ham', 'Spam'])
        ax[2].set_title('Test Data')
        ax[2].set_xticks([0, 1], ['Ham', 'Spam'])
        # for i in ax[0].containers:
        #     ax[0].bar_label(i,)
        # for i in ax[1].containers:
        #     ax[1].bar_label(i,)
        grouped_train = train.groupby(target).count()
        grouped_valid = valid.groupby(target).count()
        grouped_test = test.groupby(target).count()
        for index, row in grouped_train.iterrows():
            ax[0].text(row.name, row.text / 2, row.text, color='white', ha='center')
        for index, row in grouped_valid.iterrows():
            ax[1].text(row.name, row.text / 2, row.text, color='white', ha='center')
        for index, row in grouped_test.iterrows():
            ax[2].text(row.name, row.text / 2, row.text, color='white', ha='center')

        return fig.show()


class ActionOnDuplicates:

    def __init__(self, data):
        self.data = data

    def check_duplicates(self, subset_cols=None):
        duplicates = self.data.duplicated(subset=subset_cols, keep=False)
        duplicated_main = self.data[duplicates].sort_values(by=subset_cols)
        return duplicated_main

    def remove_duplicates_and_show(self, subset_cols=None):
        data_nodup = self.data.drop_duplicates()
        duplicates = data_nodup.duplicated(subset=subset_cols, keep=False)
        duplicated_main = data_nodup[duplicates].sort_values(by=subset_cols)
        return duplicated_main
