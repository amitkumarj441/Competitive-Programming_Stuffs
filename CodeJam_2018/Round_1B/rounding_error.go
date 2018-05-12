package main

import (
	"fmt"
)

func main() {
	var testCases int
	fmt.Scanln(&testCases)

	for testCase := 1; testCase <= testCases; testCase++ {
		var n int
		fmt.Scan(&n)
		var l int
		fmt.Scan(&l)
		fmt.Scan("\n")

		list := make([]int, l)
		for i := 0; i < l; i++ {
			fmt.Scan(&list[i])
		}
		fmt.Scan("\n")

		var total int = 0
		var currentNum int = 0
		for i := 0; i < l; i++ {
			currentNum += list[i]
			var num float64 = (float64(list[i]) / float64(n)) * 100
			var rest float64 = num - float64(int(num))
			if rest >= 0.5 {
				total += int((float64(list[i])/float64(n))*100) + 1
			} else {
				total += int((float64(list[i]) / float64(n)) * 100)
			}
		}

		var numToGo int = n - currentNum
		tmpList := make([]int, l)
		copy(tmpList, list)
		var result int = 0
		if numToGo == 0 {
			result = total
		} else if n == 10 {
			result = 100
		} else {
			var best int = total
			for j := 0; j < numToGo; j++ {
				var current int = best
				var length int = len(tmpList)
				var bestK int = 0

				for k := 0; k < length; k++ {
					var oldNum float64 = (float64(tmpList[k]) / float64(n)) * 100
					var oldRest float64 = oldNum - float64(int(oldNum))
					if oldRest >= 0.5 {
						current -= int((float64(tmpList[k])/float64(n))*100) + 1
					} else {
						current -= int((float64(tmpList[k]) / float64(n)) * 100)
					}
					var newNum float64 = (float64(tmpList[k]+1) / float64(n)) * 100
					var newRest float64 = newNum - float64(int(newNum))
					if newRest >= 0.5 {
						current += int((float64(tmpList[k]+1)/float64(n))*100) + 1
					} else {
						current += int((float64(tmpList[k]+1) / float64(n)) * 100)
					}

					if current > best {
						best = current
						if k == 0 {
							tmpList[k] += 1
						} else {
							tmpList[bestK] -= 1
							tmpList[k] += 1
						}
						bestK = k
					}

					current = total
				}

				var newNum float64 = (float64(1) / float64(n)) * 100
				var newRest float64 = newNum - float64(int(newNum))
				if newRest >= 0.5 {
					current += int((float64(1)/float64(n))*100) + 1
				} else {
					current += int((float64(1) / float64(n)) * 100)
				}

				if current > best {
					best = current
					tmpList[bestK] -= 1
					tmpList = append(tmpList, 1)
				}
				total = best
			}
			result = best
		}

		fmt.Printf("Case #%d: %d\n", testCase, result)
	}
}
