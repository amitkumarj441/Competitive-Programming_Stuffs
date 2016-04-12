// @Author: Amit Kumar Jaiswal

import java.util.Scanner;
class Karan
{
	public static void printStrings(String a)
	{
		String str = "" + a.charAt(0);
		int msi=0;
		for(int i=0;i<a.length();i++)
		{
			if(a.charAt(i)==str.charAt(msi)) 
				continue;
			else
			{
				str+=a.charAt(i);
				msi++;
			}
		}
		System.out.println(str);
	}
	public static void main(String[] args)
	{
		int k;
		String str;
		Scanner f= new Scanner(System.in);
		k=f.nextInt();
		for(int i=0;i<k;i++)
		{
			f.nextLine();
			str=f.next();
			printStrings(str);
		}
	}
}
