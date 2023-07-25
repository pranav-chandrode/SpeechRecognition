class TextProcess:
    def __init__(self):
        charaters = """
                    ' 0
                    <SPACE> 1
                    a 2
                    b 3
                    c 4
                    d 5
                    e 6
                    f 7
                    g 8
                    h 9
                    i 10
                    j 11
                    k 12
                    l 13
                    m 14
                    n 15
                    o 16
                    p 17
                    q 18
                    r 19
                    s 20
                    t 21
                    u 22
                    v 23
                    w 24
                    x 25
                    y 26
                    z 27 
                    """
        self.chat_to_ind = {}
        self.ind_to_char = {}
        for line in charaters.strip().split('\n'):
            char = line.strip().split(" ")[0]
            index = line.strip().split(" ")[1]
            self.chat_to_ind[char] = index
            self.ind_to_char[index] = char
        self.ind_to_char[1] = " "

    def text_to_int(self,text):
        int_sequence = []
        for ch in text:
            if ch == ' ':
                int_sequence.append(int(self.chat_to_ind["<SPACE>"]))
            else: int_sequence.append(int(self.chat_to_ind[ch]))
        
        return int_sequence
    
    def int_to_text(self,int_label):
        text_sequence = []
        for ind in int_label:
            text_sequence.append(self.ind_to_char[ind])

        return "".join(text_sequence).replace("<SPACE>" ," ")
