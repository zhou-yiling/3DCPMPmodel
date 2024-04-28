#ifndef  UTILITY_H
#define  UTILITY_H

void printcudaMemoryInfo();

void printTimeStamp();


//根据输入的变量类型，识别其数据类型，并用输入的数据初始化
template<typename T>
void setPara(T &var, T value)
{
	var = value;
}

#endif