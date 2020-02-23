import torch
import torch.nn as nn
import os
import gensim

class joint_embedding(nn.Module):
    def __init__(self, ori_fea_dim, joint_embedding_dim):
        super(joint_embedding, self).__init__()
        self.img_feature = nn.Linear(ori_fea_dim, joint_embedding_dim)
        self.tanh = nn.Tanh()

    def forward(self, img):
        img_feature = self.img_feature(img)
        img_feature = self.tanh(img_feature)  # [batch_size,joint_embedding_dim]
        return img_feature

class sent2vec(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(sent2vec, self).__init__()
        self.sentGRU = nn.GRU(embedding_size, hidden_size, bidirectional=True)
        self.sentVec = AttentionLayer(hidden_size)

    def forward(self, sent_embeded):
        sent_encoded, _ = self.sentGRU(sent_embeded)  # [batch_size, word_num, hidden_size*2]
        sent_vec = self.sentVec(sent_encoded)  # [batch_size,hidden_size*2]
        return sent_vec

# class sent2vec_c(nn.Module):
#     def __init__(self, embedding_size, hidden_size):
#         super(sent2vec_c, self).__init__()
#         self.sentGRU = nn.GRU(embedding_size, hidden_size, bidirectional=True)
#         self.sentVec = AttentionWithContext(hidden_size)
#
#     def forward(self, sent_embeded, context):
#         sent_encoded, _ = self.sentGRU(sent_embeded)  # [batch_size, word_num, hidden_size*2]
#         sent_vec = self.sentVec(sent_encoded, context)  # [batch_size,hidden_size*2]
#         return sent_vec

class textcnn(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(textcnn, self).__init__()
        kernel_size = [1, 2, 3, 4, 5, 6]
        channel = hidden_size * 2 // len(kernel_size)
        self.conv = nn.ModuleList([nn.Conv2d(1, channel, (K, embedding_size)) for K in kernel_size])
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch,1,word_num,embedding_size)
        x = [self.relu(conv2d(x)).squeeze() for conv2d in self.conv]  # (batch,10,word_num)
        x = [nn.functional.max_pool1d(i, i.size(2)).squeeze() for i in x]  # (batch,10)
        x = torch.cat(x, 1)  # (batch,10*kernel_num)
        return x

class SALT(nn.Module):
    def __init__(self,args):
        super(SALT,self).__init__()
        self.init_scale=args.init_scale
        self.alpha=args.alpha
        self.batch_size=args.batch_size
        self.keep_prob=args.keep_prob
        self.img_fea_dim=args.img_fea_dim
        self.joint_embedding_dim=args.joint_embedding_dim
        self.vocab_size=args.vocab_size
        self.embedding_size=args.embedding_size
        self.hidden_size=args.hidden_size
        self.max_sentence_length=args.max_sentence_length
        self.max_sentence_num=args.max_sentence_num
        self.word2vec_path=args.word2vec_path
        """
        loading embedding matrix
        """
        self.word2vec = nn.Embedding(self.vocab_size, self.embedding_size)
        self.tanh=nn.Tanh()
        self.img_feature = joint_embedding(self.img_fea_dim, self.joint_embedding_dim)
        self.titleVec=sent2vec(self.embedding_size,self.hidden_size)
        self.ingredientVec=sent2vec(self.embedding_size,self.hidden_size)

        self.contextVec=textcnn(self.embedding_size,self.hidden_size)
        self.img_feature_att=AttentionWithContext(self.joint_embedding_dim)
        self.instructionGRU=nn.GRU(self.embedding_size,self.hidden_size,bidirectional=True)
        self.instructionGRU2=nn.GRU(self.hidden_size*2,self.hidden_size,bidirectional=True)
        self.instructionVec_att=AttentionWithContext(self.joint_embedding_dim)

        self.concat_fea=nn.Linear(self.hidden_size*2,self.joint_embedding_dim)
        self.instr_fea=nn.Linear(self.hidden_size*2,self.joint_embedding_dim)
        # self.recipe_feature=joint_embedding(self.hidden_size*6,self.joint_embedding_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if os.path.exists(self.word2vec_path):
            model = gensim.models.KeyedVectors.load_word2vec_format(self.word2vec_path, binary=True)
            word_embedding = torch.FloatTensor(model.vectors)
            self.word2vec.weight.data.copy_(word_embedding)
            self.word2vec.weight.requires_grad = False

    def forward(self,inputs):
        img,title,ingredient,instruction=inputs
        img_feature=self.img_feature(img)
        img_feature=self.tanh(img_feature)
        title_word_embedded=self.word2vec(title)
        title_sent_vec=self.titleVec(title_word_embedded)
        ingredient_word_embeded=self.word2vec(ingredient)
        ingredient_sent_vec=self.ingredientVec(ingredient_word_embeded)

        context=torch.cat([title_word_embedded,ingredient_word_embeded],1)
        context=self.contextVec(context)#[128,1024]
        context=self.concat_fea(context)#
        # print(context.shape,121)
        instruction_word_embedded=self.word2vec(instruction)
        instruction_word_embedded=instruction_word_embedded.view(-1, self.max_sentence_length, self.embedding_size)
        instruction_encoded,_=self.instructionGRU(instruction_word_embedded)
        instruction_sent_vec=torch.mean(instruction_encoded,dim=1,keepdim=True)
        instruction_sent_vec=instruction_sent_vec.view(-1,self.max_sentence_num,self.hidden_size*2)
        instruction_encoded2,_=self.instructionGRU2(instruction_sent_vec)
        instruction_encoded2=self.instr_fea(instruction_encoded2)#[128,20,1024]
        # print(instruction_encoded2.shape,129)
        # print(img_feature.shape,130)
        # print()
        img_new_fea=self.img_feature_att(img_feature,context)#[img_feature 128,5,1024]
        # print(img_new_fea.shape,133)[128,1024]
        instru_new_fea=self.instructionVec_att(instruction_encoded2,context)

        img_new_fea=self.tanh(img_new_fea)
        instru_new_fea=self.tanh(instru_new_fea)

        #coattention
        # print("coattention 138")
        # print(img.shape,139)[128,5,4096]
        img_mean=torch.mean(img,dim=1,keepdim=True)
        img_mean=img_mean.squeeze()

        img_feature_mean=self.img_feature(img_mean)
        img_feature_mean=self.tanh(img_feature_mean)
        # print(instruction_encoded2.shape,148)[128,20,1024]
        # print(img_feature_mean.shape, 149)#[128,1024]
        instru_co_fea=self.instructionVec_att(instruction_encoded2,img_feature_mean)
        img_coatt_fea=self.img_feature_att(img_feature,instru_co_fea)
        instru_co_fea=self.tanh(instru_co_fea)
        img_coatt_fea=self.tanh(img_coatt_fea)
        # print(self.alpha,154) 0.4
        instru_new_fea = instru_new_fea * self.alpha
        img_new_fea = img_new_fea * self.alpha
        instru_co_fea = instru_co_fea * (1 - self.alpha)
        img_coatt_fea = img_coatt_fea * (1 - self.alpha)

        instru_co_super_fea = torch.cat([instru_new_fea, instru_co_fea], 1)
        # print(instru_co_super_fea.shape,113)
        img_co_super_fea = torch.cat([img_new_fea, img_coatt_fea], 1)

        return img_co_super_fea,instru_co_super_fea

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.u = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.u_context = nn.Linear(hidden_size * 2, 1, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):  # [batch,word_num,hidden_size*2]
        batch_size = x.size(0)
        x1 = x.view(-1, self.hidden_size * 2)  # [batch*word_num,hidden_size*2]
        x1 = self.u(x1)
        u = self.tanh(x1)
        alpha = self.u_context(u)  # [batch_size*word_num,1]
        alpha = alpha.view(batch_size, -1)  # [batch_size,word_num]
        alpha = self.softmax(alpha)
        output = torch.sum(torch.mul(alpha.unsqueeze(2), x), dim=1)  # [batch_size,hidden_size*2]
        return output

class AttentionWithContext(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionWithContext, self).__init__()
        self.hidden_size = hidden_size
        self.u = nn.Linear(hidden_size , hidden_size )
        self.u_context = nn.Linear(hidden_size , hidden_size)
        self.tanh = nn.Tanh()
        self.W = nn.Linear(hidden_size , 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, context):  # [batch,word_num,hidden_size*2] [batch,hidden_size*2]
        # print(x.shape,191)#[128,5,1024]
        # print(context.shape,192)#[128,1024]
        batch_size = x.size(0)
        x1 = x.view(-1, self.hidden_size)  # [batch*word_num,hidden_size*2]
        repeat = x1.size(0) // context.size(0)
        context = context.repeat(repeat, 1).view(-1, self.hidden_size )
        x1 = self.u(x1)
        x2 = self.u_context(context)
        # print(x1.shape,202)[640,1024]
        # print(x2.shape,203)[640,1024]
        u = self.tanh(x1 + x2)  # [batch*word_num,hidden_size*2]
        alpha = self.W(u)
        alpha = alpha.view(batch_size, -1)  # [batch_size,word_num]
        alpha = self.softmax(alpha)
        output = torch.sum(torch.mul(alpha.unsqueeze(2), x), dim=1)  # [batch_size,hidden_size*2]
        return output
