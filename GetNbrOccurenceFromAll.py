from projectHelper import getIndexFromDonneeTest
from projectHelper import getIndexFromTP
from projectHelper import getIndexFromTweet
from IndexBuilder import IndexBuilder
import re

index = getIndexFromDonneeTest(getIndexFromTP(getIndexFromTweet()))
index.SaveNbrOccurence(salt="allWord_")