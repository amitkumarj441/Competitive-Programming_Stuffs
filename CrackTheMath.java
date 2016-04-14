// Author: Amit Kumar Jaiswal

import java.util.Scanner;
class CrackTheMath 
{
	public static String crackTheMath(String x,String y)
	{
		String r="";
		for(int i=0;i<x.length();i++)
		{
			r+=((int)x.charAt(i)^(int)y.charAt(i));
		}
		return r;
	}
	public static void main(String[] args)
	{
		Scanner in=new Scanner(System.in);
		int z=in.nextInt();
		for(int i=0;i<z;i++)
		{
			String x = in.next();
			String y = in.next();
			System.out.println(crackTheMath(x,y));
		}
	}
}
