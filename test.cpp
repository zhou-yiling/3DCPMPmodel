#include <iostream>




templete<typename T>
void setPara(T &var, T vaule)
{
	*var = value;
}

int main()
{
    int a = 0;
    setPara(a, 1);
    std::cout << a << std::endl;
    return 0;
}