import numpy as np
from Encoder import EncoderArchitecture
from Decoder import DecoderArchitecture
from Vocublary_matrix import VocabMatrix
from inputembeeding import InputEmbedding
from LinearAndSoftmax import LinearSoftmaxOutput

class Transformer:
    def __init__(self,blocks=1):
        self.blocks=blocks
        self.embeded=VocabMatrix()
        self.vec_dim=self.embeded.em_dim
        self.vocab_size=len(self.embeded.word_to_token)
        self.token_to_word={}
        for key,val in self.embeded.word_to_token.items():
            self.token_to_word[val]=key
            
        self.inp_mat=InputEmbedding()
        self.out_mat=InputEmbedding()
        
        self.EncoderArray=[]
        self.DecoderArray=[]
        for i in range(self.blocks):
            self.EncoderArray.append(EncoderArchitecture())
            self.DecoderArray.append(DecoderArchitecture())
            
        self.LinearSoftResult=LinearSoftmaxOutput(self.vec_dim,self.vocab_size)
        
    def loss_function(self,yPred,tar):
        win_size,v_size=yPred.shape
        self.yPred=yPred
        self.y_Onehot=np.zeros_like(yPred)
        text_split=tar.split.lower()
        unknown=self.embeded.word_to_token['unk']
        token_id=[]
        for i in text_split:
            token_id.append(self.embeded.word_to_token.get(i,unknown))
            
        if len(token_id)<win_size:
            token_id+=[0]*(win_size-len(token_id))
        else:
            token_id=token_id[:win_size]
            
        for i,t in enumerate(token_id):
            self.y_Onehot[i,t]=1.0
        
        loss=-np.sum(self.y_Onehot*np.log(self.yPred))/win_size
        return loss        
        
            
def RunTransformer(self,inp_text,dec_input_text):
    X_enc=self.inp_mat.initializeEmbedding(inp_text,self.emd.word_to_token,self.emd.embedding_matrix)
    for i in range(self.block):
        X_enc=self.EncoderArray[i].InitializeEncoder(X_enc)
    
    X_dec=self.out_mat.initializeEmbedding(dec_input_text,self.emd.word_to_token,self.emd.embedding_matrix)
    for i in range(self.block):
        X_dec=self.DecoderArray[i].initializeDecoder(X_enc,X_dec)
    out=self.LinearSoft.Linear(X_dec)
    return out

def Backpropagate(self,lr):
    d_soft=self.y_pred-self.y_onehot
    d_dec=self.LinearSoft.backpropagate(d_soft)
    d_enc_total=np.zeros_like(self.EncoderArray[-1].Encoder_res)
    d_out=d_dec
    for i in range(self.block-1,-1,-1):
        d_enc_i,d_out=self.DecoderArray[i].backpropagate(d_out)
        d_enc_total+=d_enc_i
    
    for i in range(self.block-1,-1,-1):
        d_enc_total=self.EncoderArray[i].backpropagate(d_enc_total)
    d_emb_inp=self.inp_mat.backpropagate(d_enc_total)
    d_emb_out=self.out_mat.backpropagate(d_out)
    self.LinearSoft.update(lr)
    
    for dec in self.DecoderArray:
        dec.update(lr)
    for enc in self.EncoderArray:
        enc.update(lr)
    self.emd.update(lr,d_emb_inp+d_emb_out)

def Train(self,train_data,epochs=2000,lr=0.05):
    for epoch in range(epochs):
        total_loss=0.0
        for sample in train_data:
            summary_words=sample["summary"].lower().split()
            dec_inp="<start> "+" ".join(summary_words[:-1])+" "+"<end>"
            dec_tar=" ".join(summary_words)
            y_pred=self.RunTransformer(sample["reviews"],dec_inp)
            loss=self.Loss_function(y_pred,dec_tar)
            self.Backpropagate(lr)
            total_loss+=loss
        
        avg_loss=total_loss/len(train_data)
        if epoch%100==0:
            print(f"Epoch {epoch} | Avg Loss: {avg_loss:.6f}")
        if avg_loss<0.001:
            print(f"Overfitting  at epoch {epoch}")
            break

def Predict(self,review,max_len=10):
    inp_emb=self.inp_mat.initializeEmbedding(review,self.emd.word_to_token,self.emd.embedding_matrix)
    enc=inp_emb
    for i in range(self.block):
        enc=self.EncoderArray[i].InitializeEncoder(enc)
    start_id=self.emd.word_to_token.get("<start>",0)
    pad_id=self.emd.word_to_token.get("<pad>",0)
    end_id=self.emd.word_to_token.get("<end>",None)
    current_ids=[start_id]
    for p in range(max_len):
        words=[self.token_to_word.get(i,"<unk>") for i in current_ids]
        dec_text=" ".join(words)
        dec_emb=self.out_mat.initializeEmbedding(dec_text,self.emd.word_to_token,self.emd.embedding_matrix)
        dec=dec_emb
        for i in range(self.block):
            dec=self.DecoderArray[i].initializeDecoder(enc,dec)
        probs=self.LinearSoft.Linear(dec)
        next_id=int(np.argmax(probs[-1]))
        if next_id==pad_id or (end_id is not None and next_id==end_id):
            break
        current_ids.append(next_id)
    words=[self.token_to_word.get(i,"<unk>") for i in current_ids]
    if words and words[0]=="<start>":
        words=words[1:]
    return " ".join(words)

if __name__ == "__main__":
    transformer = Transformer(blocks=1)

    train_data = [
        {"reviews": "sound quality good battery excellent easy connect case solid",
         "summary": "great sound and battery"},
        {"reviews": "poor charging issue after two days unworthy product bad quality",
         "summary": "charging problem and poor quality"},
        {"reviews": "amazing bass noise cancellation works perfectly battery life long",
         "summary": "excellent sound and battery"},
        {"reviews": "left earbud stopped working after month charger issue return bad",
         "summary": "earbud and charger defective"}
    ]

    transformer.Train(train_data, epochs=2000, lr=0.01)

    print("\nPredictions after training:")
    for d in train_data:
        print("Review :", d["reviews"])
        print("Pred   :", transformer.Predict(d["reviews"], max_len=8))
        print("Target :", d["summary"])
        print("------")

        
