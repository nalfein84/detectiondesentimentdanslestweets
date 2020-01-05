
import operator

class IndexBuilder:

    def __init__(self):
        self.wordslist = {}
        self.elemID = 0
        self.wordID = 0

    def AddElem(self, elem):
        mots = elem.split()
        for mot in mots:
            if self.wordslist.has_key(mot):
                self.wordslist[mot][0] = 1 + int(self.wordslist[mot][0])
                self.wordslist[mot][1].append(self.elemID)
            else:
                self.wordslist[mot] = [1, [self.elemID], int(self.wordID)]
                self.wordID += 1
        self.elemID += 1

    def SaveNbrOccurence(self, salt=""):
        with open(salt + 'nbrOccurence.txt', 'w') as fn:  # open file safely in append mode
            dico_trie = sorted(self.wordslist.iteritems(), reverse=True,
                            key=operator.itemgetter(1))
            for row in dico_trie:
                fn.write(row[0]+':'+str(row[1][0])+'\n')

    def GetNombreOccurence(self):
        dico_trie = sorted(self.wordslist.iteritems(), reverse=True, key=operator.itemgetter(1))
        return dico_trie

    def SaveCorrelation(self, salt=""):
        with open(salt + 'matriceCorrelation.txt', 'w') as fn:
            for key in self.wordslist.keys():
                fn.write(key+':' + str(self.wordslist[key][1]) + '\n')

    def GetValues(self):
        return self.wordslist
