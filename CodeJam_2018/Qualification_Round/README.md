# Problems

## P1: Saving the Universe Again

An alien robot is threatening the universe, using a beam that will destroy all algorithms knowledge. We have to stop it!

Fortunately, we understand how the robot works. It starts off with a beam with a strength of 1, and it will run a program that is a series of instructions, which will be executed one at a time, in left to right order. Each instruction is of one of the following two types:

- C (for "charge"): Double the beam's strength.
- S (for "shoot"): Shoot the beam, doing damage equal to the beam's current strength.

For example, if the robot's program is SCCSSC, the robot will do the following when the program runs:

- Shoot the beam, doing 1 damage.
- Charge the beam, doubling the beam's strength to 2.
- Charge the beam, doubling the beam's strength to 4.
- Shoot the beam, doing 4 damage.
- Shoot the beam, doing 4 damage.
- Charge the beam, increasing the beam's strength to 8.

In that case, the program would do a total of 9 damage.

The universe's top algorithmists have developed a shield that can withstand a maximum total of D damage. But the robot's current program might do more damage than that when it runs.

The President of the Universe has volunteered to fly into space to hack the robot's program before the robot runs it. The only way the President can hack (without the robot noticing) is by swapping two adjacent instructions. For example, the President could hack the above program once by swapping the third and fourth instructions to make it SCSCSC. This would reduce the total damage to 7. Then, for example, the president could hack the program again to make it SCSSCC, reducing the damage to 5, and so on.

To prevent the robot from getting too suspicious, the President does not want to hack too many times. What is this smallest possible number of hacks which will ensure that the program does no more than D total damage, if it is possible to do so?

### Input
The first line of the input gives the number of test cases, T. T test cases follow. Each consists of one line containing an integer D and a string P: the maximum total damage our shield can withstand, and the robot's program.

### Output
For each test case, output one line containing Case #x: y, where x is the test case number (starting from 1) and y is either the minimum number of hacks needed to accomplish the goal, or IMPOSSIBLE if it is not possible.

### Limits
1 ≤ T ≤ 100.

1 ≤ D ≤ 109.

2 ≤ length of P ≤ 30.

Every character in P is either C or S.

Time limit: 20 seconds per test set.

Memory limit: 1GB.

```Test set 1 (Visible)```

The robot's program contains either zero or one C characters.

```Test set 2 (Hidden)```

No additional restrictions to the Limits section.


# P2: Troble Sort

Deep in Code Jam's secret algorithm labs, we devote countless hours to wrestling with one of the most complex problems of our time: efficiently sorting a list of integers into non-decreasing order. We have taken a careful look at the classic bubble sort algorithm, and we are pleased to announce a new variant.

The basic operation of the standard bubble sort algorithm is to examine a pair of adjacent numbers, and reverse that pair if the left number is larger than the right number. But our algorithm examines a group of three adjacent numbers, and if the leftmost number is larger than the rightmost number, it reverses that entire group. Because our algorithm is a "triplet bubble sort", we have named it Trouble Sort for short.

```
  TroubleSort(L): // L is a 0-indexed list of integers
    let done := false
    while not done:
      done = true
      for i := 0; i < len(L)-2; i++:
        if L[i] > L[i+2]:
          done = false
          reverse the sublist from L[i] to L[i+2], inclusive
          
```

For example, for L = 5 6 6 4 3, Trouble Sort would proceed as follows:

- First pass:
  - inspect 5 6 6, do nothing: 5 6 6 4 3
  - inspect 6 6 4, see that 6 > 4, reverse the triplet: 5 4 6 6 3
  - inspect 6 6 3, see that 6 > 3, reverse the triplet: 5 4 3 6 6

- Second pass:
  - inspect 5 4 3, see that 5 > 3, reverse the triplet: 3 4 5 6 6
  - inspect 4 5 6, do nothing: 3 4 5 6 6
  - inspect 5 6 6, do nothing: 3 4 5 6 6

- Then the third pass inspects the three triplets and does nothing, so the algorithm terminates.

We were looking forward to presenting Trouble Sort at the Special Interest Group in Sorting conference in Hawaii, but one of our interns has just pointed out a problem: it is possible that Trouble Sort does not correctly sort the list! Consider the list 8 9 7, for example.

We need your help with some further research. Given a list of N integers, determine whether Trouble Sort will successfully sort the list into non-decreasing order. If it will not, find the index (counting starting from 0) of the first sorting error after the algorithm has finished: that is, the first value that is larger than the value that comes directly after it when the algorithm is done.

### Input
The first line of the input gives the number of test cases, T. T test cases follow. Each test case consists of two lines: one line with an integer N, the number of values in the list, and then another line with N integers Vi, the list of values.

### Output
For each test case, output one line containing Case #x: y, where x is the test case number (starting from 1) and y is OK if Trouble Sort correctly sorts the list, or the index (counting starting from 0) of the first sorting error, as described above.

