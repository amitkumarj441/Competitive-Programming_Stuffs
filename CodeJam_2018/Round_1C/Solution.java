import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.*;

public class Solution {

    public static void main(String[] args) {
        Scanner in = new Scanner(new BufferedReader(new InputStreamReader(System.in)));
        int t = in.nextInt();  
        for (int i = 1; i <= t; ++i) {
            int wordsNumber = in.nextInt();
            int wordLength = in.nextInt();
            Set<String> words = new HashSet<>(wordsNumber);
            for (int wordId = 0; wordId < wordsNumber; ++wordId) {
                words.add(in.next());
            }
            System.out.println("Case #" + i + ": " + prepareAnswer(getNewWord(words, wordLength), words));
        }
    }

    private static String getNewWord(Set<String> words, int wordLength) {
        if (words.size() == 1) {
            return null;
        }
        StringBuilder newWord = new StringBuilder(words.iterator().next());
        List<Set<Character>> letterMap = getLetterMap(words, wordLength);

        genWord(newWord, 0, wordLength, letterMap, words);

        return newWord.toString();
    }

    private static boolean genWord(StringBuilder wordBuilder, int letterPos, int wordLength,
                                List<Set<Character>> letterMap, Set<String> forbiddenWords) {
        if (letterPos == wordLength) {
            return forbiddenWords.contains(wordBuilder.toString());
        }

        Iterator<Character> avalibleLetters = letterMap.get(letterPos).iterator();
        do {
            if (!avalibleLetters.hasNext()) {
                return true;
            }
            Character currLetter = avalibleLetters.next();
            wordBuilder.setCharAt(letterPos, currLetter);
        } while (genWord(wordBuilder, letterPos + 1, wordLength, letterMap, forbiddenWords));

        return false;
    }

    private static List<Set<Character>> getLetterMap(Set<String> words, int wordLength) {
        List<Set<Character>> map = new ArrayList<>(wordLength);
        for (int i = 0; i < wordLength; ++i) {
            map.add(new HashSet<>());
        }
        for (String word : words) {
            int i = 0;
            for (char c : word.toCharArray()) {
                map.get(i).add(c);
                i++;
            }
        }
        return map;
    }

    private static String prepareAnswer(String newWord, Set<String> words) {
        return ((newWord == null) || newWord.isEmpty() || words.contains(newWord)) ? "-" : newWord;
    }
}
