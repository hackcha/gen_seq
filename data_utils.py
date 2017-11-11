import pickle
import numpy as np

PAD_ID = 0
GO_ID = 1 #翻译的开始
EOS_ID = 2 #句子结束
UNK_ID = 3
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]
# max_vocabulary_size = 10000

def pad_data(terms_list , max_len , pad_pre = False):
    if max_len is None:
        max_len = 0
        for terms in terms_list:
            if len(terms) > max_len:
                max_len = len(terms)
    new_terms_list = []
    for terms in terms_list:
        pad_len = max_len-len(terms)
        if pad_len > 0:
            if pad_pre:
                new_terms = [PAD_ID]*pad_len + terms
            else:
                new_terms = terms + [PAD_ID]*pad_len
        else:
            new_terms = terms[-max_len:]
        new_terms_list.append(new_terms)
    return new_terms_list

def load_vocab( vocabulary_path):
    vocab_dict = {}
    vocab_res = {}
    vid = 0
    with open(vocabulary_path, mode="r", encoding='utf-8') as vocab_file:
        for w in vocab_file:
            vocab_dict[w.strip()] = vid
            vocab_res[vid] = w.strip()
            vid += 1
    return vocab_dict, vocab_res

class DataConverter(object):
    def __init__(self):
        self.vocab = dict( )

    def build_vocab(self, lines):
        counter = 0
        for line in lines:
            counter += 1
            if counter % 100000 == 0:
                print("processing line %d" % counter)
            for w in line:
                if w==' ' or w == '' or w=='\t' or w=='\n' or w=='\r':
                    continue
                if w in self.vocab:
                    self.vocab[w] += 1
                else:
                    self.vocab[w] = 1

    def save_vocab(self, vocabulary_path):
        vocab_list = _START_VOCAB+sorted(self.vocab, key=self.vocab.get, reverse=True)
        # if len(vocab_list) > max_vocabulary_size:
        #     vocab_list = vocab_list[:max_vocabulary_size]
        with open(vocabulary_path, mode="w", encoding='utf-8') as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + "\n")
        print('save vocab done.')



    def convert(self, data_dir, fileout='./data.pkl'):
        #qtais_tab.txt 只有三个字段
        files = ['ming.all', 'qing.all', 'qsc_tab.txt', 'qss_tab.txt', 'qtais_tab.txt', 'qts_tab.txt', 'yuan.all']
        for file in files:
            print('file : {}'.format( file ) )
            with open( data_dir+file, 'r', encoding='utf-8') as fin:
                all_lines = []
                for line in fin:
                    terms = line.split('\t')
                    if len(terms[-1]) > 50:
                        continue
                    all_lines.append( terms[-1].strip() )
                self.build_vocab( all_lines )
        vocab_path = './vocab.txt'
        self.save_vocab( vocab_path )
        converted = []
        self.vocab_dict, self.vocab_res = load_vocab( vocab_path )
        print('start padded.')
        max_len = 0
        for file in files:
            print('file : {}'.format( file ) )
            with open( data_dir+file, 'r', encoding='utf-8') as fin:
                for line in fin:
                    terms = line.split('\t')
                    poet = terms[-1].strip()
                    if len(terms[-1]) > 50:
                        continue
                    term_ids = []
                    if len(poet) > max_len:
                        max_len = len(poet)
                    term_ids.append( GO_ID )
                    for w in poet:
                        if w in self.vocab_dict:
                            term_ids.append(self.vocab_dict[w])
                        else:
                            term_ids.append(UNK_ID)
                    #古诗结束
                    term_ids.append( EOS_ID )
                    converted.append( term_ids )
        print('max len: {}'.format( max_len ) )
        # max_len = 50
        padded = pad_data( converted, max_len+2 )
        print('padded done.')
        padded = np.array( padded, dtype='int32' )
        pickle.dump(padded, open(fileout, 'wb'))
        print('done.')



def load_data( fn = './data.pkl' ):
    data = pickle.load( open(fn,'rb' ) )
    return data

if __name__ == '__main__':
    data_dir = 'D:/dataset/rnnpg_data_emnlp-2014/raw_poem_all/'
    converter = DataConverter( )
    converter.convert( data_dir )



