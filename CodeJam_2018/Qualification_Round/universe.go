package main

import (
	"fmt"
	"strings"
)

func main() {
	var testCases int
	fmt.Scanln(&testCases)

	for testCase := 1; testCase <= testCases; testCase++ {
		var shield int
		var instr string
		fmt.Scan(&shield, &instr)

		var length = len(instr)
		var charge = 1
		var damage = 0
		var swap = 0
		var shoots int

		for i := 0; i < length; i++ {
			if instr[i] == 'C' {
				charge *= 2
			} else {
				shoots += 1
				damage += charge
			}
		}

		if shoots > shield {
			fmt.Printf("Case #%d: IMPOSSIBLE\n", testCase)
		} else {
			for damage > shield {
				var idx = strings.LastIndex(instr, "CS")
				if idx == length-2 {
					instr = instr[:idx] + "SC"
				} else if idx == 0 {
					instr = "SC" + instr[(idx+2):]
				} else if idx == -1 {
					break
				} else {
					instr = instr[:idx] + "SC" + instr[(idx+2):]
				}

				var currentCharge = 1
				for i := 0; i < idx; i++ {
					if instr[i] == 'C' {
						currentCharge *= 2
					}
				}
				damage -= currentCharge

				swap += 1
			}
			fmt.Printf("Case #%d: %d\n", testCase, swap)
		}
	}
}
