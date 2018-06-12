//Author: Amit Kumar Jaiswal

import java.util.*;
import static java.lang.System.*;
import static java.lang.Integer.*;
 
public class Main {
public static void main(String[] args) {
        Scanner sc = new Scanner(in);
        int N = sc.nextInt();
        String [] st = new String[N];
        for (int i = 0; i < N; i++) {
            st[i] = sc.next();
        }
        int Q = sc.nextInt();
        for (int i = 0; i < Q; i++) {
            int R = sc.nextInt();
            String P = sc.next();
 
            int resIndex = 0, maxLCP = MIN_VALUE;
            for (int j = 0; j < R; j++) {
                int LCP = 0;
                String s = st[j];
 
                for (int k = 0; k < s.length() && k < P.length(); k++) {
 
                    if (s.charAt(k) == P.charAt(k))
                        LCP += 1;
                    else
                        break;
                }
                if (LCP > maxLCP)
                {
                    maxLCP = LCP;
                    resIndex = j;
                }
                else if (LCP == maxLCP)
                {
                    if (s.compareTo(st[resIndex]) <= 0)
                        resIndex = j;
                }
            }
            out.println(st[resIndex]);
        }
    }
} 
