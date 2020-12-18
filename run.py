import os
from pyswip import Prolog
import nltk
import collections
import summary


def readFile(path="./news",nowText="TEST.txt"):
    with open(path+"/"+nowText,"r") as f:
        lines=[]
        sentence=[]
        eng_sentence=[]
        eng_sentence_txt=[]
        while True:
            line=f.readline().replace('\r\n',"").replace(',',"")
            if not line: break
            lines.append(line)
        for line in lines:
            result=line.replace('\n',"").split(".")
            if type(result) is list: 
                sentence=sentence+result
            else: 
                sentence.append(line.split("."))
        for s in sentence:
            if s==" " or s=="\n": continue
            else:
                if s+"."==".": continue
                eng_sentence_txt.append("["+s.lower().replace(" ",",")+"].")
                eng_sentence.append(s.lower()+".")
    return eng_sentence,eng_sentence_txt

def updateLexicon(eng_sentence):
    def build_dictionary():
        dictionary = {}
        for sent in eng_sentence:
            pos_tags = nltk.pos_tag(nltk.word_tokenize(sent))
            for tag in pos_tags:
                value = tag[0]
                pos = tag[1]
                dictionary[value] = pos
        return dictionary
    
    pos_dict = build_dictionary() # List of eng_sentence
    
    lex=[]
    
    for _,(key,item) in enumerate(pos_dict.items()):
        # check there is the word in wordList.pickle
        if key in [".","is","not","are","mia","vincent"]:
            continue
        
        if item=="JJ" or item=="JJR" or item=="JJS" or item=="RB" or item=="RBR" or item=="RBS" or item=="VBN" or item=="PRP":
            lex.append("lexEntry(adj,[symbol:%s,syntax:[%s]])."%(key,key))
        elif item=="VBG":
            lex.append("lexEntry(pi,[symbol:%s,syntax:[%s]])."%(key,key))
            lex.append("lexEntry(adj,[symbol:%s,syntax:[%s]])."%(key,key))
        elif item=="NN" or item=="NNS":
            lex.append("lexEntry(noun,[symbol:%s,syntax:[%s]])."%(key,key))
        elif item=="NNP" or item=="NNPS" or item=="FW" or item=="LS": # with special case
            lex.append("lexEntry(pn,[symbol:%s,syntax:[%s]])."%(key,key))
        elif item=="VB" or item=="VBP" or item=="VBD" or item=="RP":
            lex.append("lexEntry(tv,[symbol:%s,syntax:[%s],inf:fin,num:sg])."%(key,key))
            lex.append("lexEntry(tv,[symbol:%s,syntax:[%s],inf:fin,num:pl])."%(key,key))
            lex.append("lexEntry(tv,[symbol:%s,syntax:[%s],inf:inf,num:sg])."%(key,key))
            lex.append("lexEntry(tv,[symbol:%s,syntax:[%s],inf:inf,num:pl])."%(key,key))
            lex.append("lexEntry(iv,[symbol:%s,syntax:[%s],inf:fin,num:sg])."%(key,key))
            lex.append("lexEntry(iv,[symbol:%s,syntax:[%s],inf:fin,num:pl])."%(key,key))
            lex.append("lexEntry(iv,[symbol:%s,syntax:[%s],inf:inf,num:sg])."%(key,key))
            lex.append("lexEntry(iv,[symbol:%s,syntax:[%s],inf:inf,num:pl])."%(key,key))
        elif item=="VBZ":
            lex.append("lexEntry(tv,[symbol:%s,syntax:[%s],inf:fin,num:sg])."%(key,key))
            lex.append("lexEntry(iv,[symbol:%s,syntax:[%s],inf:fin,num:sg])."%(key,key))
        elif item=="WP":
            lex.append("lexEntry(qnp,[symbol:thing,syntax:[%s],mood:int,type:wh])."%(key))
        elif item=="WRB":
            lex.append("lexEntry(qnp,[symbol:place,syntax:[%s],mood:int,type:wh])."%(key))
        elif item=="WDT" or item=="WP$":
            lex.append("lexEntry(det,[syntax:[%s],mood:int,type:wh])."%(key))
        elif item=="IN" or item=="TO":
            lex.append("lexEntry(prep,[symbol:%s,syntax:[%s]])."%(key,key))
        elif item=="CC":
            lex.append("lexEntry(coord,[syntax:[%s],type:conj])."%(key))
        elif item=="DT":
            if key not in ["a","the","every","which"]:
                lex.append("lexEntry(det,[syntax:[%s],mood:int,type:wh])."%(key))
        else:
            lex.append("lexEntry(noun,[symbol:%s,syntax:[%s]])."%(key,key)) 
        
    #APPEND WordList into englishLexicon.pl
    with open("englishLexicon.pl","a") as f:
        f.write("\n")
        for lexentry in lex:
            f.write(lexentry+"\n")


def run():
    print("======= input your command  ===========")
    print("quit/setFolder/showList/pointIn/pointOut/isPoint/question/summary")
    print("=======================================")
    textList=dict() # {path1:{nowText1:Prolog,nowText2:Prolog},path2:{nowText1:Prolog,nowText2:Prolog}}
    nowText=""
    path=""
    
    while True:
        print("> ",end="")
        sentence=input().split()
        command = sentence[0]
        
        if command=="quit":
            break
            
        elif command=="setFolder":
            if not os.path.isdir(sentence[1]):
                print("plz, write down a correct directory")
                continue
            path=sentence[1]

        elif command=="showList":
            if path=="":
                print("plz, write down path")
                continue
            files=os.listdir(path)
            for f in files:
                if "txt" in f:
                    print(f)
        elif command=="pointOut":
            nowText=""
        elif command=="isPoint":
            if nowText!="":
                print("nowText : %s \n"%(nowText))
            else:
                print("there is not nowText")
        elif command=="pointIn":
            if not os.path.isfile(path+"/"+sentence[1]):
                print("plz, write down a correct file"+path+sentence[1])
                continue
            nowText=sentence[1]
            
            eng_sentence,eng_sentence_txt=readFile(path,nowText)
            
            with open("TMP.txt","w") as f:
                for i in range(len(eng_sentence_txt)):
                    if i==len(eng_sentence_txt)-1:
                        f.write(eng_sentence_txt[i])
                    else:
                        f.write(eng_sentence_txt[i]+"\n")
            
            if path not in textList.keys():
                textList[path]=dict()
            if nowText not in textList[path].keys():
                textList[path][nowText]=Prolog()
                
                
                ### append wordList #### 
                with open("../wordList.txt","r") as f:
                    lines=[]
                    while True:
                        line=f.readline().replace('\r\n',"")
                        if not line: break
                        lines.append(line)
                if nowText in lines or nowText+"\n" in lines:
                    pass
                else:    
                    with open("../wordList.txt","a") as f:
                        f.write(nowText+"\n")
                        updateLexicon(eng_sentence)
                
            
        elif command=="question":
            if nowText!="":
                print("====== question mode ==========")
                print("If you want to stop the question, you write down `bye`")
                # focus on one file
                textList[path][nowText].consult("bot.pl")
                print("===== end of question mode =====")
            else:
                print("Write down nowText!")
                continue
                
        elif command=="summary":
            result = summary.summary(nowText, path+"/"+nowText)
            print(result)

        else:
            print("plz, write down a correct command")
            continue
            
            
if __name__=="__main__":
    run()