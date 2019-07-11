
# coding: utf-8

# ## 1.  Importing the libraries

# In[ ]:


import pandas as pd
from pathlib import Path


# In[ ]:


from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import WordEmbeddings, FlairEmbeddings
from flair.data_fetcher import NLPTaskDataFetcher


# ## 2. class Dataloader  ( From FLAIR )
# #### For some reason it can't be imported using  : from flair.datasets import DataLoader

# In[1]:


import torch.utils.data.dataloader
from torch.utils.data.dataset import Subset, ConcatDataset


# In[2]:


class DataLoader(torch.utils.data.dataloader.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=8,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):

        # in certain cases, multi-CPU data loading makes no sense and slows
        # everything down. For this reason, we detect if a dataset is in-memory:
        # if so, num_workers is set to 0 for faster processing
        flair_dataset = dataset
        while True:
            if type(flair_dataset) is Subset:
                flair_dataset = flair_dataset.dataset
            elif type(flair_dataset) is ConcatDataset:
                flair_dataset = flair_dataset.datasets[0]
            else:
                break

        if type(flair_dataset) is list:
            num_workers = 0
        elif isinstance(flair_dataset, FlairDataset) and flair_dataset.is_in_memory():
            num_workers = 0

        super(DataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=list,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


# ### 3.  Loading the datasets and preprocessing it for FLAIR

# In[ ]:


df_train = pd.read_csv('/.../train-train.csv' )

df_train_flair = df_train["Class"]
df_train_flair = df_train_flair.to_frame()
df_train_flair["Sentence"] = df_train["Sentence"]

df_train_flair.shape


# In[ ]:


df_test = pd.read_csv('/.../train-dev.csv' )

df_test_flair = df_test["Class"]
df_test_flair = df_test_flair.to_frame()
df_test_flair["Sentence"] = df_test["Sentence"]
df_test_flair = df_test_flair[:50]
df_test_flair.shape


# In[ ]:


df_train_flair['Class'] = '__label__' + df_train_flair['Class'].astype(str)
df_test_flair['Class'] = '__label__' + df_test_flair['Class'].astype(str)


# In[ ]:


df_train_flair.to_csv('/.../train.csv', sep='\t', index = False, header = False)
df_test_flair.to_csv('/.../test.csv', sep='\t', index = False, header = False)


# In[ ]:


corpus = NLPTaskDataFetcher.load_classification_corpus(Path('/content/drive/My Drive/Offenseval_Datasets/folder/'), test_file='test.csv', train_file = 'train.csv')


# ### 4. This is the module embedding.py 
# ##### Class edited : class DocumentLSTMEmbeddings()

# In[4]:


import os
import re
import logging
from abc import abstractmethod
from collections import Counter
from pathlib import Path
from typing import List, Union, Dict
import gensim
import numpy as np
import torch
from bpemb import BPEmb
from deprecated import deprecated

from pytorch_pretrained_bert import (
    BertTokenizer,
    BertModel,
    TransfoXLTokenizer,
    TransfoXLModel,
    OpenAIGPTModel,
    OpenAIGPTTokenizer,
)


# In[6]:


from pytorch_pretrained_bert.modeling_openai import (
    PRETRAINED_MODEL_ARCHIVE_MAP as OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_MAP,
)

from pytorch_pretrained_bert.modeling_transfo_xl import (
    PRETRAINED_MODEL_ARCHIVE_MAP as TRANSFORMER_XL_PRETRAINED_MODEL_ARCHIVE_MAP,
)

import flair
from flair.data import Corpus
from flair.data import Sentence, Corpus, Token


# In[ ]:


class Embeddings(torch.nn.Module):
    """Abstract base class for all embeddings. Every new type of embedding must implement these methods."""

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        pass

    @property
    @abstractmethod
    def embedding_type(self) -> str:
        pass

    def embed(self, sentences: Union[Sentence, List[Sentence]]) -> List[Sentence]:
        """Add embeddings to all words in a list of sentences. If embeddings are already added, updates only if embeddings
        are non-static."""

        # if only one sentence is passed, convert to list of sentence
        if type(sentences) is Sentence:
            sentences = [sentences]

        everything_embedded: bool = True

        if self.embedding_type == "word-level":
            for sentence in sentences:
                for token in sentence.tokens:
                    if self.name not in token._embeddings.keys():
                        everything_embedded = False
        else:
            for sentence in sentences:
                if self.name not in sentence._embeddings.keys():
                    everything_embedded = False

        if not everything_embedded or not self.static_embeddings:
            self._add_embeddings_internal(sentences)

        return sentences

    @abstractmethod
    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        """Private method for adding embeddings to all words in a list of sentences."""
        pass



class TokenEmbeddings(Embeddings):
    """Abstract base class for all token-level embeddings. Ever new type of word embedding must implement these methods."""

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        pass

    @property
    def embedding_type(self) -> str:
        return "word-level"


class StackedEmbeddings(TokenEmbeddings):
    """A stack of embeddings, used if you need to combine several different embedding types."""

    def __init__(self, embeddings: List[TokenEmbeddings], detach: bool = True):
        """The constructor takes a list of embeddings to be combined."""
        super().__init__()
        
        self.embeddings = embeddings

        # IMPORTANT: add embeddings as torch modules
        for i, embedding in enumerate(embeddings):
            self.add_module("list_embedding_{}".format(i), embedding)

        self.detach: bool = detach
        self.name: str = "Stack"
        self.static_embeddings: bool = True

        self.__embedding_type: str = embeddings[0].embedding_type
          
        self.__embedding_length: int = 0
        for embedding in embeddings:
            self.__embedding_length += embedding.embedding_length
            
        
    def embed(
        self, sentences: Union[Sentence, List[Sentence]], static_embeddings: bool = True
    ):
        # if only one sentence is passed, convert to list of sentence
        if type(sentences) is Sentence:
            sentences = [sentences]

        for embedding in self.embeddings:
            embedding.embed(sentences)

    @property
    def embedding_type(self) -> str:
        return self.__embedding_type

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for embedding in self.embeddings:
            embedding._add_embeddings_internal(sentences)

        return sentences

    def __str__(self):
        return f'StackedEmbeddings [{",".join([str(e) for e in self.embeddings])}]'

      

class DocumentEmbeddings(Embeddings):
    """Abstract base class for all document-level embeddings. Ever new type of document embedding must implement these methods."""

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        pass

    @property
    def embedding_type(self) -> str:
        return "sentence-level"      
      
class DocumentLSTMEmbeddings(DocumentEmbeddings):
    def __init__(
        self,
        embeddings: List[TokenEmbeddings],
        hidden_size=128,
        rnn_layers=1,
        reproject_words: bool = True,
        reproject_words_dimension: int = None,
        bidirectional: bool = True,
        dropout: float = 0.5,
        word_dropout: float = 0.0,
        locked_dropout: float = 0.0,
    ):
        """The constructor takes a list of embeddings to be combined.
        :param embeddings: a list of token embeddings
        :param hidden_size: the number of hidden states in the lstm
        :param rnn_layers: the number of layers for the lstm
        :param reproject_words: boolean value, indicating whether to reproject the token embeddings in a separate linear
        layer before putting them into the lstm or not
        :param reproject_words_dimension: output dimension of reprojecting token embeddings. If None the same output
        dimension as before will be taken.
        :param bidirectional: boolean value, indicating whether to use a bidirectional lstm or not
        :param dropout: the dropout value to be used
        :param word_dropout: the word dropout value to be used, if 0.0 word dropout is not used
        :param locked_dropout: the locked dropout value to be used, if 0.0 locked dropout is not used
        """
        super().__init__()

        self.embeddings: StackedEmbeddings = StackedEmbeddings( embeddings = embeddings )

        self.reproject_words = reproject_words
        self.bidirectional = bidirectional

        self.length_of_all_token_embeddings: int = self.embeddings.embedding_length

        self.name = "document_lstm"
        self.static_embeddings = False

        self.__embedding_length: int = hidden_size
        if self.bidirectional:
            self.__embedding_length *= 4

        self.embeddings_dimension: int = self.length_of_all_token_embeddings
    
        if self.reproject_words and reproject_words_dimension is not None:
            self.embeddings_dimension = reproject_words_dimension

        # bidirectional LSTM on top of embedding layer
        self.word_reprojection_map = torch.nn.Linear(
            self.length_of_all_token_embeddings, self.embeddings_dimension
        )
        self.rnn = torch.nn.GRU(
            self.embeddings_dimension,
            hidden_size,
            num_layers=rnn_layers,
            bidirectional=self.bidirectional,
        )
        
        
        # dropouts
        if locked_dropout > 0.0:
            self.dropout: torch.nn.Module = LockedDropout(locked_dropout)
        else:
            self.dropout = torch.nn.Dropout(dropout)

        self.use_word_dropout: bool = word_dropout > 0.0
        if self.use_word_dropout:
            self.word_dropout = WordDropout(word_dropout)

        torch.nn.init.xavier_uniform_(self.word_reprojection_map.weight)

        self.to(flair.device)

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def embed(self, sentences: Union[List[Sentence], Sentence]):
        """Add embeddings to all sentences in the given list of sentences. If embeddings are already added, update
         only if embeddings are non-static."""

        if type(sentences) is Sentence:
            sentences = [sentences]

        self.rnn.zero_grad()

        sentences.sort(key=lambda x: len(x), reverse=True)

        self.embeddings.embed(sentences)

        # first, sort sentences by number of tokens
        longest_token_sequence_in_batch: int = len(sentences[0])
      
        all_sentence_tensors = []
        lengths: List[int] = []

        # go through each sentence in batch
        for i, sentence in enumerate(sentences):

            lengths.append(len(sentence.tokens))

            word_embeddings = []

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                word_embeddings.append(token.get_embedding().unsqueeze(0))
            
        
            # PADDING: pad shorter sentences out
            for add in range(longest_token_sequence_in_batch - len(sentence.tokens)):
                word_embeddings.append(
                    torch.zeros(
                        self.length_of_all_token_embeddings, dtype=torch.float
                    ).unsqueeze(0)
                )
        
            word_embeddings_tensor = torch.cat(word_embeddings, 0).to(flair.device)
           

            sentence_states = word_embeddings_tensor

            """
            1. The normal function is to send the word embeddings for the entire batch through a linear layer and then LSTM and
               then further take the representations from there for each sentence. The dimension of the word embedding tensor for 
               each sentence was [( hidden_size )]. This would  make the input for the CNN layer only a 2d vector. 
            
            2.Right now the sentence has an embedding_tensor of the shape [( max_sentence_len , embedding_length )]
            
            3.The same word embedding tensor has been attached each sentence so that the input to the CNN layer cna be a 
              3d vector.
            
            """
            
            ### ADDING IT TO THE SENTENCE DIRECTLY
            
            sentence.set_embedding(self.name, word_embeddings_tensor)
            
            print("SIZE OF THE EMBEDDING ADDED TO THE CURRENT SENTENCE IS : ")
            print( word_embeddings_tensor.size() )
            
            
            """
            # ADD TO SENTENCE LIST: add the representation
            all_sentence_tensors.append(sentence_states.unsqueeze(1))
            #all_sentence_tensors.append(sentence_states)
            
            print("6 . All_sentence_tensors just one element  : ")
            print(all_sentence_tensors[0].size())
            #print(all_sentence_tensors[0])
            """

            
        """    
        # --------------------------------------------------------------------
        # GET REPRESENTATION FOR ENTIRE BATCH
        # --------------------------------------------------------------------
        
        print("THE REPRESENTATION OF THE ENTIRE BATCH : ")
        print()
        sentence_tensor = torch.cat(all_sentence_tensors, 1)
        print()
        print(sentence_tensor.size())
        print()

        # --------------------------------------------------------------------
        # FF PART
        # --------------------------------------------------------------------
        # use word dropout if set
        if self.use_word_dropout:
            sentence_tensor = self.word_dropout(sentence_tensor)

        if self.reproject_words:
            sentence_tensor = self.word_reprojection_map(sentence_tensor)
            
        print("8 . AFTER REPROJECT_WORDS : ")
        print(sentence_tensor.size())

        sentence_tensor = self.dropout(sentence_tensor)

        packed = torch.nn.utils.rnn.pack_padded_sequence(sentence_tensor, lengths)
       

        self.rnn.flatten_parameters()

        lstm_out, hidden = self.rnn(packed)

        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
        print("OUTPUTS :")
        print(outputs.size())
        print()
        print("8 . AFTER RNN_PAD_PACKED : OUTPUTS ")
        print( outputs.size() )
        
        print("8.1 OUTPUTS AFTER SQUEEZE() : ")
        print( outputs.squeeze(1).size() )
        print(outputs.squeeze(1))

        outputs = self.dropout(outputs)
        
        print()
        print("9 . AFTER DROPOUT : OUTPUTS ")
        print( outputs.size() )

        print("***length***")
        print(lengths)
        # --------------------------------------------------------------------
        # EXTRACT EMBEDDINGS FROM LSTM
        # --------------------------------------------------------------------
        
        for sentence_no, length in enumerate(lengths):
            print(" 9. sentence_no :")
            print(sentence_no)
            print()
            print(" 9. length :")
            print(length)
            print()
            
            #last_rep = outputs[length - 1, sentence_no]
            #print("LAST_REP : ")
            #print(last_rep.size())
            #print(last_rep)
            
            embedding = outputs.squeeze(1)
            #if self.bidirectional:
                #first_rep = outputs[0, sentence_no]
                #print("FIRST_REP : ")
                #print(first_rep.size())
                #embedding = torch.cat([first_rep, last_rep], 0)
                #print("CONCAT FIRST AND LAST_REP :")
                #print(embedding.size())
            print("EMBEDDING BEING ATTACHED TO THE SENTENCE : ")
            print(embedding.size())
            sentence = sentences[sentence_no]
            sentence.set_embedding(self.name, outputs.squeeze(1))
        """
        
        
    def _add_embeddings_internal(self, sentences: List[Sentence]):
        pass


      
      
      


# ## 5.  Creating embeddings for our corpus : 

# In[ ]:


flair_forward_embedding = FlairEmbeddings('multi-forward')
flair_backward_embedding = FlairEmbeddings('multi-backward')


# In[ ]:


word_embeddings = [ flair_forward_embedding ,  flair_backward_embedding  ]

document_embeddings = DocumentLSTMEmbeddings(word_embeddings, hidden_size=512, 
reproject_words = False , reproject_words_dimension=256)


# ### 6. class Textclassifier ( changes have been made in _init_() and forward() )
# 

# In[7]:


from flair.data import Dictionary, Sentence, Label
from flair.file_utils import cached_path
from flair.training_utils import (
    convert_labels_to_one_hot,
    clear_embeddings,
    Metric,
    Result,
)

from typing import List, Union


# In[ ]:


class TextClassifier(flair.nn.Model):
    """
    Text Classification Model
    The model takes word embeddings, puts them into an RNN to obtain a text representation, and puts the
    text representation in the end into a linear layer to get the actual class label.
    The model can handle single and multi class data sets.
    """

    def __init__(
        self,
        df_train , 
        document_embeddings: flair.embeddings.DocumentEmbeddings,
        label_dictionary: Dictionary,
        multi_label: bool = None,
        multi_label_threshold: float = 0.5,
        
    ):

        super(TextClassifier, self).__init__()

        
        
        self.document_embeddings: flair.embeddings.DocumentRNNEmbeddings = document_embeddings
        self.label_dictionary: Dictionary = label_dictionary
        kernel_size = [ 5 , 6 ]
            
        """ 
        Calculating the maximum_length_of_the_sentence
        """
        
        train_dev_sentences_for_prediction = self.test_to_sentences(  df_train_flair )
        train_dev_sentences_for_prediction.sort(key=lambda x: len(x), reverse=True)
        longest_token_sequence_in_batch: int = len(train_dev_sentences_for_prediction[0])
        self.max_sen_len = longest_token_sequence_in_batch
         
        if multi_label is not None:
            self.multi_label = multi_label
        else:
            self.multi_label = self.label_dictionary.multi_label

        self.multi_label_threshold = multi_label_threshold

        """
        The CNN layer : 
        in_channel : embedding_length
        
        As of now self.document_embeddings.embedding_length does not give the actual embedding_size of the final 
        embedding_tensor. Will try out something to accomadate this.  
        """
        self.conv1 = nn.Sequential(
            nn.Conv1d( in_channels = 4096 , out_channels = 100 , kernel_size = kernel_size[0] ) ,      
            nn.ReLU(),
            nn.MaxPool1d(self.max_sen_len - kernel_size[0] + 1)
        )
        
    
        # self.decoder = nn.Linear( num_channels * len( kernel_size ) , len(self.label_dictionary() ))    
        self.decoder = nn.Linear( 1 * 100 , len(self.label_dictionary)
        )

        self._init_weights()

        if self.multi_label:
            self.loss_function = nn.BCEWithLogitsLoss()
        else:
            self.loss_function = nn.CrossEntropyLoss()

        # auto-spawn on GPU if available
        self.to(flair.device)
        
        
        
    def test_to_sentences( self, df_test_flair ):
        train_dev_sents_list = [  x for x in df_test_flair['Sentence'] ]
        print("Sentences list created : ")
        train_dev_sentences_for_prediction = []

        for i in train_dev_sents_list : 
            train_dev_sentences_for_prediction.append(Sentence(i))
    
        return train_dev_sentences_for_prediction    
    
    
    def _init_weights(self):
        ##initialising weights for the linear layer
        nn.init.xavier_uniform_(self.decoder.weight)
     
        
   
    def _get_state_dict(self):
        print("Model's state_dict:")
        
        """
        To check each layer of the model
        """
        for param_tensor in self.state_dict():
            print(param_tensor, "\t", self.state_dict()[param_tensor].size())
       
    
        model_state = {
            "state_dict": self.state_dict(),
            "document_embeddings": self.document_embeddings,
            "label_dictionary": self.label_dictionary,
            "multi_label": self.multi_label,
        }
        return model_state    
    
    def forward(self, sentences) -> List[List[float]]:
        
        self.document_embeddings.embed(sentences)

        text_embedding_list = [
            sentence.get_embedding().unsqueeze(0) for sentence in sentences
        ]
       
        text_embedding_tensor = torch.cat(text_embedding_list, 0).to(flair.device)
        
        embedded_sent = text_embedding_tensor.permute(0,2,1)
        # embedded_sent.size() = ( batch_size ,  embedding_size , max_sentence_length )    
    
        conv_out1 = self.conv1( embedded_sent ).squeeze(2)
        label_scores = self.decoder( conv_out1 )

        return label_scores

      
    def _init_model_with_state_dict(state):

        model = TextClassifier(
            document_embeddings=state["document_embeddings"],
            label_dictionary=state["label_dictionary"],
            multi_label=state["multi_label"],
        )

        model.load_state_dict(state["state_dict"])
        return model

    def forward_loss(self, sentences: Union[List[Sentence], Sentence]) -> torch.tensor:
        scores = self.forward(sentences)
        return self._calculate_loss(scores, sentences)

    def forward_labels_and_loss(
        self, sentences: Union[Sentence, List[Sentence]]
    ) -> (List[List[Label]], torch.tensor):
        scores = self.forward(sentences)
        labels = self._obtain_labels(scores)
        loss = self._calculate_loss(scores, sentences)
        return labels, loss

    def predict(
        self,
        sentences: Union[Sentence, List[Sentence]],
        mini_batch_size: int = 32,
        multi_class_prob: bool = False,
    ) -> List[Sentence]:
        """
        Predicts the class labels for the given sentences. The labels are directly added to the sentences.
        :param sentences: list of sentences
        :param mini_batch_size: mini batch size to use
        :param multi_class_prob : return probability for all class for multiclass
        :return: the list of sentences containing the labels
        """
        with torch.no_grad():
            if type(sentences) is Sentence:
                sentences = [sentences]

            filtered_sentences = self._filter_empty_sentences(sentences)

            batches = [
                filtered_sentences[x : x + mini_batch_size]
                for x in range(0, len(filtered_sentences), mini_batch_size)
            ]

            for batch in batches:
                scores = self.forward(batch)
                predicted_labels = self._obtain_labels(
                    scores, predict_prob=multi_class_prob
                )

                for (sentence, labels) in zip(batch, predicted_labels):
                    sentence.labels = labels

                clear_embeddings(batch)

            return sentences

    def evaluate(
        self,
        sentences: List[Sentence],
        eval_mini_batch_size: int = 32,
        embeddings_in_memory: bool = False,
        out_path: Path = None,
        num_workers: int = 8,
    ) -> (Result, float):

        with torch.no_grad():
            eval_loss = 0

            batch_loader = DataLoader(
                sentences,
                batch_size=eval_mini_batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

            metric = Metric("Evaluation")

            lines: List[str] = []
            batch_count: int = 0
            for batch in batch_loader:

                batch_count += 1

                labels, loss = self.forward_labels_and_loss(batch)

                clear_embeddings(
                    batch, also_clear_word_embeddings=not embeddings_in_memory
                )

                eval_loss += loss

                sentences_for_batch = [sent.to_plain_string() for sent in batch]
                confidences_for_batch = [
                    [label.score for label in sent_labels] for sent_labels in labels
                ]
                predictions_for_batch = [
                    [label.value for label in sent_labels] for sent_labels in labels
                ]
                true_values_for_batch = [
                    sentence.get_label_names() for sentence in batch
                ]
                available_labels = self.label_dictionary.get_items()

                for sentence, confidence, prediction, true_value in zip(
                    sentences_for_batch,
                    confidences_for_batch,
                    predictions_for_batch,
                    true_values_for_batch,
                ):
                    eval_line = "{}\t{}\t{}\t{}\n".format(
                        sentence, true_value, prediction, confidence
                    )
                    lines.append(eval_line)

                for predictions_for_sentence, true_values_for_sentence in zip(
                    predictions_for_batch, true_values_for_batch
                ):

                    for label in available_labels:
                        if (
                            label in predictions_for_sentence
                            and label in true_values_for_sentence
                        ):
                            metric.add_tp(label)
                        elif (
                            label in predictions_for_sentence
                            and label not in true_values_for_sentence
                        ):
                            metric.add_fp(label)
                        elif (
                            label not in predictions_for_sentence
                            and label in true_values_for_sentence
                        ):
                            metric.add_fn(label)
                        elif (
                            label not in predictions_for_sentence
                            and label not in true_values_for_sentence
                        ):
                            metric.add_tn(label)

            eval_loss /= batch_count

            detailed_result = (
                f"\nMICRO_AVG: acc {metric.micro_avg_accuracy()} - f1-score {metric.micro_avg_f_score()}"
                f"\nMACRO_AVG: acc {metric.macro_avg_accuracy()} - f1-score {metric.macro_avg_f_score()}"
            )
            for class_name in metric.get_classes():
                detailed_result += (
                    f"\n{class_name:<10} tp: {metric.get_tp(class_name)} - fp: {metric.get_fp(class_name)} - "
                    f"fn: {metric.get_fn(class_name)} - tn: {metric.get_tn(class_name)} - precision: "
                    f"{metric.precision(class_name):.4f} - recall: {metric.recall(class_name):.4f} - "
                    f"accuracy: {metric.accuracy(class_name):.4f} - f1-score: "
                    f"{metric.f_score(class_name):.4f}"
                )

            result = Result(
                main_score=metric.micro_avg_f_score(),
                log_line=f"{metric.precision()}\t{metric.recall()}\t{metric.micro_avg_f_score()}",
                log_header="PRECISION\tRECALL\tF1",
                detailed_results=detailed_result,
            )

            if out_path is not None:
                with open(out_path, "w", encoding="utf-8") as outfile:
                    outfile.write("".join(lines))

            return result, eval_loss

    @staticmethod
    def _filter_empty_sentences(sentences: List[Sentence]) -> List[Sentence]:
        filtered_sentences = [sentence for sentence in sentences if sentence.tokens]
        if len(sentences) != len(filtered_sentences):
            log.warning(
                "Ignore {} sentence(s) with no tokens.".format(
                    len(sentences) - len(filtered_sentences)
                )
            )
        return filtered_sentences

    def _calculate_loss(
        self, scores: torch.tensor, sentences: List[Sentence]
    ) -> torch.tensor:
        """
        Calculates the loss.
        :param scores: the prediction scores from the model
        :param sentences: list of sentences
        :return: loss value
        """
        if self.multi_label:
            return self._calculate_multi_label_loss(scores, sentences)

        return self._calculate_single_label_loss(scores, sentences)

    def _obtain_labels(
        self, scores: List[List[float]], predict_prob: bool = False
    ) -> List[List[Label]]:
        """
        Predicts the labels of sentences.
        :param scores: the prediction scores from the model
        :return: list of predicted labels
        """

        if self.multi_label:
            return [self._get_multi_label(s) for s in scores]

        elif predict_prob:
            return [self._predict_label_prob(s) for s in scores]

        return [self._get_single_label(s) for s in scores]

    def _get_multi_label(self, label_scores) -> List[Label]:
        labels = []

        sigmoid = torch.nn.Sigmoid()

        results = list(map(lambda x: sigmoid(x), label_scores))
        for idx, conf in enumerate(results):
            if conf > self.multi_label_threshold:
                label = self.label_dictionary.get_item_for_index(idx)
                labels.append(Label(label, conf.item()))

        return labels

    def _get_single_label(self, label_scores) -> List[Label]:
        softmax = torch.nn.functional.softmax(label_scores, dim=0)
        conf, idx = torch.max(softmax, 0)
        label = self.label_dictionary.get_item_for_index(idx.item())

        return [Label(label, conf.item())]

    def _predict_label_prob(self, label_scores) -> List[Label]:
        softmax = torch.nn.functional.softmax(label_scores, dim=0)
        label_probs = []
        for idx, conf in enumerate(softmax):
            label = self.label_dictionary.get_item_for_index(idx)
            label_probs.append(Label(label, conf.item()))
        return label_probs

    def _calculate_multi_label_loss(
        self, label_scores, sentences: List[Sentence]
    ) -> float:
        return self.loss_function(label_scores, self._labels_to_one_hot(sentences))

    def _calculate_single_label_loss(
        self, label_scores, sentences: List[Sentence]
    ) -> float:
        return self.loss_function(label_scores, self._labels_to_indices(sentences))

    def _labels_to_one_hot(self, sentences: List[Sentence]):
        label_list = [sentence.get_label_names() for sentence in sentences]
        one_hot = convert_labels_to_one_hot(label_list, self.label_dictionary)
        one_hot = [torch.FloatTensor(l).unsqueeze(0) for l in one_hot]
        one_hot = torch.cat(one_hot, 0).to(flair.device)
        return one_hot

    def _labels_to_indices(self, sentences: List[Sentence]):
        indices = [
            torch.LongTensor(
                [
                    self.label_dictionary.get_idx_for_item(label.value)
                    for label in sentence.labels
                ]
            )
            for sentence in sentences
        ]

        vec = torch.cat(indices, 0).to(flair.device)

        return vec

    def _fetch_model(model_name) -> str:

        model_map = {}
        aws_resource_path = (
            "https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/models-v0.4"
        )

        model_map["de-offensive-language"] = "/".join(
            [
                aws_resource_path,
                "TEXT-CLASSIFICATION_germ-eval-2018_task-1",
                "germ-eval-2018-task-1.pt",
            ]
        )

        model_map["en-sentiment"] = "/".join(
            [aws_resource_path, "TEXT-CLASSIFICATION_imdb", "imdb.pt"]
        )

        cache_dir = Path("models")
        if model_name in model_map:
            model_name = cached_path(model_map[model_name], cache_dir=cache_dir)

        return model_name


# In[ ]:


classifier = TextClassifier( df_train_flair , document_embeddings, 
label_dictionary = corpus.make_label_dictionary(), multi_label = False )


# ### 7. Training 
# #### class ModelTrainer() : 
#      

# In[ ]:


from pathlib import Path
from typing import List, Union

import datetime

from torch.optim.sgd import SGD
from torch.utils.data.dataset import ConcatDataset

import flair
import flair.nn
from flair.data import Sentence, MultiCorpus, Corpus
#from flair.datasets import DataLoader

from flair.training_utils import (
    init_output_file,
    WeightExtractor,
    clear_embeddings,
    EvaluationMetric,
    log_line,
    add_file_handler,
    Result,
)
from flair.optim import *

log = logging.getLogger("flair")

class ModelTrainer:
    def __init__(
        self,
        model: flair.nn.Model,
        corpus: Corpus,
        optimizer: Optimizer = SGD,
        epoch: int = 0,
        loss: float = 10000.0,
        optimizer_state: dict = None,
        scheduler_state: dict = None,
    ):
        self.model: flair.nn.Model = model
        self.corpus: Corpus = corpus
        self.optimizer: Optimizer = optimizer
        self.epoch: int = epoch
        self.loss: float = loss
        self.scheduler_state: dict = scheduler_state
        self.optimizer_state: dict = optimizer_state

    def train(
        self,
        base_path: Union[Path, str],
        evaluation_metric: EvaluationMetric = EvaluationMetric.MICRO_F1_SCORE,
        learning_rate: float = 0.1,
        mini_batch_size: int = 32,
        eval_mini_batch_size: int = None,
        max_epochs: int = 100,
        anneal_factor: float = 0.5,
        patience: int = 3,
        train_with_dev: bool = False,
        monitor_train: bool = False,
        monitor_test: bool = False,
        embeddings_in_memory: bool = True,
        checkpoint: bool = False,
        save_final_model: bool = True,
        anneal_with_restarts: bool = False,
        shuffle: bool = True,
        param_selection_mode: bool = False,
        num_workers: int = 8,
        sampler=None,
        **kwargs,
    ) -> dict:

        if eval_mini_batch_size is None:
            eval_mini_batch_size = mini_batch_size

        # cast string to Path
        if type(base_path) is str:
            base_path = Path(base_path)

        log_handler = add_file_handler(log, base_path / "training.log")

        log_line(log)
        log.info(f'Model: "{self.model}"')
        log_line(log)
        log.info(f'Corpus: "{self.corpus}"')
        log_line(log)
        log.info("Parameters:")
        log.info(f' - learning_rate: "{learning_rate}"')
        log.info(f' - mini_batch_size: "{mini_batch_size}"')
        log.info(f' - patience: "{patience}"')
        log.info(f' - anneal_factor: "{anneal_factor}"')
        log.info(f' - max_epochs: "{max_epochs}"')
        log.info(f' - shuffle: "{shuffle}"')
        log.info(f' - train_with_dev: "{train_with_dev}"')
        log_line(log)
        log.info(f'Model training base path: "{base_path}"')
        log_line(log)
        log.info(f"Evaluation method: {evaluation_metric.name}")

        # determine what splits (train, dev, test) to evaluate and log
        log_train = True if monitor_train else False
        log_test = (
            True
            if (not param_selection_mode and self.corpus.test and monitor_test)
            else False
        )
        log_dev = True if not train_with_dev else False

        # prepare loss logging file and set up header
        loss_txt = init_output_file(base_path, "loss.tsv")

        weight_extractor = WeightExtractor(base_path)

        optimizer = self.optimizer(self.model.parameters(), lr=learning_rate, **kwargs)
        if self.optimizer_state is not None:
            optimizer.load_state_dict(self.optimizer_state)

        # minimize training loss if training with dev data, else maximize dev score
        anneal_mode = "min" if train_with_dev else "max"

        if isinstance(optimizer, (AdamW, SGDW)):
            scheduler = ReduceLRWDOnPlateau(
                optimizer,
                factor=anneal_factor,
                patience=patience,
                mode=anneal_mode,
                verbose=True,
            )
        else:
            scheduler = ReduceLROnPlateau(
                optimizer,
                factor=anneal_factor,
                patience=patience,
                mode=anneal_mode,
                verbose=True,
            )
        if self.scheduler_state is not None:
            scheduler.load_state_dict(self.scheduler_state)

        train_data = self.corpus.train

        # if training also uses dev data, include in training set
        if train_with_dev:
            train_data = ConcatDataset([self.corpus.train, self.corpus.dev])

        if sampler is not None:
            sampler = sampler(train_data)

        dev_score_history = []
        dev_loss_history = []
        train_loss_history = []
        
       

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            previous_learning_rate = learning_rate

            for epoch in range(0 + self.epoch, max_epochs + self.epoch):
                log_line(log)

                # get new learning rate
                for group in optimizer.param_groups:
                    learning_rate = group["lr"]

                # reload last best model if annealing with restarts is enabled
                if (
                    learning_rate != previous_learning_rate
                    and anneal_with_restarts
                    and (base_path / "best-model.pt").exists()
                ):
                    log.info("resetting to best model")
                    self.model.load(base_path / "best-model.pt")

                previous_learning_rate = learning_rate

                # stop training if learning rate becomes too small
                if learning_rate < 0.0001:
                    log_line(log)
                    log.info("learning rate too small - quitting training!")
                    log_line(log)
                    break

                batch_loader = DataLoader(
                    train_data,
                    batch_size=mini_batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    sampler=sampler,
                )

                self.model.train()

                train_loss: float = 0

                seen_batches = 0
                total_number_of_batches = len(batch_loader)
                
                modulo = max(1, int(total_number_of_batches / 10))

                # process mini-batches
                for batch_no, batch in enumerate(batch_loader):
                    
                    loss = self.model.forward_loss(batch)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    optimizer.step()

                    seen_batches += 1
                    train_loss += loss.item()

                    clear_embeddings(
                        batch, also_clear_word_embeddings=not embeddings_in_memory
                    )

                    if batch_no % modulo == 0:
                        log.info(
                            f"epoch {epoch + 1} - iter {batch_no}/{total_number_of_batches} - loss "
                            f"{train_loss / seen_batches:.8f}"
                        )
                        iteration = epoch * total_number_of_batches + batch_no
                        if not param_selection_mode:
                            weight_extractor.extract_weights(
                                self.model.state_dict(), iteration
                            )

                train_loss /= seen_batches

                self.model.eval()

                log_line(log)
                log.info(
                    f"EPOCH {epoch + 1} done: loss {train_loss:.4f} - lr {learning_rate:.4f}"
                )

                # anneal against train loss if training with dev, otherwise anneal against dev score
                current_score = train_loss

                # evaluate on train / dev / test split depending on training settings
                result_line: str = ""

                if log_train:
                    train_eval_result, train_loss = self.model.evaluate(
                        self.corpus.train,
                        eval_mini_batch_size,
                        embeddings_in_memory,
                        num_workers=num_workers,
                    )
                    result_line += f"\t{train_eval_result.log_line}"

                if log_dev:
                    dev_eval_result, dev_loss = self.model.evaluate(
                        self.corpus.dev,
                        eval_mini_batch_size,
                        embeddings_in_memory,
                        num_workers=num_workers,
                    )
                    result_line += f"\t{dev_loss}\t{dev_eval_result.log_line}"
                    log.info(
                        f"DEV : loss {dev_loss} - score {dev_eval_result.main_score}"
                    )
                    # calculate scores using dev data if available
                    # append dev score to score history
                    dev_score_history.append(dev_eval_result.main_score)
                    dev_loss_history.append(dev_loss)

                    current_score = dev_eval_result.main_score

                if log_test:
                    test_eval_result, test_loss = self.model.evaluate(
                        self.corpus.test,
                        eval_mini_batch_size,
                        embeddings_in_memory,
                        base_path / "test.tsv",
                        num_workers=num_workers,
                    )
                    result_line += f"\t{test_loss}\t{test_eval_result.log_line}"
                    log.info(
                        f"TEST : loss {test_loss} - score {test_eval_result.main_score}"
                    )

                # determine learning rate annealing through scheduler
                scheduler.step(current_score)

                train_loss_history.append(train_loss)

                # determine bad epoch number
                try:
                    bad_epochs = scheduler.num_bad_epochs
                except:
                    bad_epochs = 0
                for group in optimizer.param_groups:
                    new_learning_rate = group["lr"]
                if new_learning_rate != previous_learning_rate:
                    bad_epochs = patience + 1

                # log bad epochs
                log.info(f"BAD EPOCHS (no improvement): {bad_epochs}")

                # output log file
                with open(loss_txt, "a") as f:

                    # make headers on first epoch
                    if epoch == 0:
                        f.write(
                            f"EPOCH\tTIMESTAMP\tBAD_EPOCHS\tLEARNING_RATE\tTRAIN_LOSS"
                        )

                        if log_train:
                            f.write(
                                "\tTRAIN_"
                                + "\tTRAIN_".join(
                                    train_eval_result.log_header.split("\t")
                                )
                            )
                        if log_dev:
                            f.write(
                                "\tDEV_LOSS\tDEV_"
                                + "\tDEV_".join(dev_eval_result.log_header.split("\t"))
                            )
                        if log_test:
                            f.write(
                                "\tTEST_LOSS\tTEST_"
                                + "\tTEST_".join(
                                    test_eval_result.log_header.split("\t")
                                )
                            )

                    f.write(
                        f"\n{epoch}\t{datetime.datetime.now():%H:%M:%S}\t{bad_epochs}\t{learning_rate:.4f}\t{train_loss}"
                    )
                    f.write(result_line)

                    
                    
                """
                1. As of now it cannot save the best model so this has been commented. Since it cannot save the best model it
                   will not work on the test set . To save the best model and retrieve it 
                   for testing on the test dataset , i will have to save the TextClassifier() seperately as and import it . 
                   Hopefully it should work then . 
                
                """    
                    
                """     
                # if checkpoint is enable, save model at each epoch
                if checkpoint and not param_selection_mode:
                    self.model.save_checkpoint(
                        base_path / "checkpoint.pt",
                        optimizer.state_dict(),
                        scheduler.state_dict(),
                        epoch + 1,
                        train_loss,
                    )

                # if we use dev data, remember best model based on dev evaluation score
                if (
                    not train_with_dev
                    and not param_selection_mode
                    and current_score == scheduler.best
                ):
                    self.model.save(base_path / "best-model.pt")
                 """
                
            """    
            # if we do not use dev data for model selection, save final model
            if save_final_model and not param_selection_mode:
                self.model.save(base_path / "final-model.pt")
            """    

        except KeyboardInterrupt:
            log_line(log)
            log.info("Exiting from training early.")
            if not param_selection_mode:
                log.info("Saving model ...")
                self.model.save(base_path / "final-model.pt")
                log.info("Done.")

        """        
        # test best model if test data is present
        if self.corpus.test:
            final_score = self.final_test(
                base_path,
                embeddings_in_memory,
                evaluation_metric,
                eval_mini_batch_size,
                num_workers,
            )
        else:
            final_score = 0
            log.info("Test data not provided setting final score to 0")

        log.removeHandler(log_handler)

        return {
            "test_score": final_score,
            "dev_score_history": dev_score_history,
            "train_loss_history": train_loss_history,
            "dev_loss_history": dev_loss_history,
        }

        """
    def final_test(
        self,
        base_path: Path,
        embeddings_in_memory: bool,
        evaluation_metric: EvaluationMetric,
        eval_mini_batch_size: int,
        num_workers: int = 8,
    ):

        log_line(log)
        log.info("Testing using best model ...")

        self.model.eval()

        if (base_path / "best-model.pt").exists():
            self.model = self.model.load(base_path / "best-model.pt")

        test_results, test_loss = self.model.evaluate(
            self.corpus.test,
            eval_mini_batch_size=eval_mini_batch_size,
            embeddings_in_memory=embeddings_in_memory,
            out_path=base_path / "test.tsv",
            num_workers=num_workers,
        )

        test_results: Result = test_results
        log.info(test_results.log_line)
        log.info(test_results.detailed_results)
        log_line(log)

        # if we are training over multiple datasets, do evaluation for each
        if type(self.corpus) is MultiCorpus:
            for subcorpus in self.corpus.corpora:
                log_line(log)
                self.model.evaluate(
                    subcorpus.test,
                    eval_mini_batch_size,
                    embeddings_in_memory,
                    base_path / f"{subcorpus.name}-test.tsv",
                )

        # get and return the final test score of best model
        final_score = test_results.main_score

        return final_score

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint, corpus: Corpus, optimizer: Optimizer = SGD
    ):
        return ModelTrainer(
            checkpoint["model"],
            corpus,
            optimizer,
            epoch=checkpoint["epoch"],
            loss=checkpoint["loss"],
            optimizer_state=checkpoint["optimizer_state_dict"],
            scheduler_state=checkpoint["scheduler_state_dict"],
        )

    def find_learning_rate(
        self,
        base_path: Union[Path, str],
        file_name: str = "learning_rate.tsv",
        start_learning_rate: float = 1e-7,
        end_learning_rate: float = 10,
        iterations: int = 100,
        mini_batch_size: int = 32,
        stop_early: bool = True,
        smoothing_factor: float = 0.98,
        **kwargs,
    ) -> Path:
        best_loss = None
        moving_avg_loss = 0

        # cast string to Path
        if type(base_path) is str:
            base_path = Path(base_path)
        learning_rate_tsv = init_output_file(base_path, file_name)

        with open(learning_rate_tsv, "a") as f:
            f.write("ITERATION\tTIMESTAMP\tLEARNING_RATE\tTRAIN_LOSS\n")

        optimizer = self.optimizer(
            self.model.parameters(), lr=start_learning_rate, **kwargs
        )

        train_data = self.corpus.train

        batch_loader = DataLoader(train_data, batch_size=mini_batch_size, shuffle=True)

        scheduler = ExpAnnealLR(optimizer, end_learning_rate, iterations)

        model_state = self.model.state_dict()
        model_device = next(self.model.parameters()).device
        self.model.train()

        for itr, batch in enumerate(batch_loader):
            loss = self.model.forward_loss(batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            learning_rate = scheduler.get_lr()[0]

            loss_item = loss.item()
            if itr == 0:
                best_loss = loss_item
            else:
                if smoothing_factor > 0:
                    moving_avg_loss = (
                        smoothing_factor * moving_avg_loss
                        + (1 - smoothing_factor) * loss_item
                    )
                    loss_item = moving_avg_loss / (1 - smoothing_factor ** (itr + 1))
                if loss_item < best_loss:
                    best_loss = loss

            if stop_early and (loss_item > 4 * best_loss or torch.isnan(loss)):
                log_line(log)
                log.info("loss diverged - stopping early!")
                break

            if itr > iterations:
                break

            with open(learning_rate_tsv, "a") as f:
                f.write(
                    f"{itr}\t{datetime.datetime.now():%H:%M:%S}\t{learning_rate}\t{loss_item}\n"
                )

        self.model.load_state_dict(model_state)
        self.model.to(model_device)

        log_line(log)
        log.info(f"learning rate finder finished - plot {learning_rate_tsv}")
        log_line(log)

        return Path(learning_rate_tsv)


# In[ ]:


trainer = ModelTrainer(classifier, corpus)
trainer.train('./', max_epochs = 5)

