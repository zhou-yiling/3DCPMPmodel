#define CONTACTANGLE	//接触角模型 _设置相应的流场
#define PTCALCFIVEPOINT //五点差分计算化学势
//#define PTCALCSEVENPOINT //七点差分计算化学势
//#define SIMPLECHEMBOUNDARY	 //简单化学势边界
#define COMPLEXCHEMBOUNDARY	 //带梯度的化学势边界

//#define DROPLETSPLASH

#ifdef WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif
#include <cassert>
#include <string>
#include <cstdlib>
#include <sstream> 
#include <fstream> 
#include <iomanip>
#include <iostream> 
#include <algorithm>
#include <cuda_runtime.h>
#include <curand.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include <math.h> 
#include "helper_cuda.h"
#include "ordered_tec.h"
#include "Common.h"
using namespace std;
using namespace ORDERED_TEC;


//******************************************SchemeNumber*******************************************
// const int DROPLETSPLASH_FIELD_INIT = 0;
// const int CONTACTANGLE_FIELD_INIT = 1;


//******************************************ErrorCheck*******************************************
int err_den = false, err_distribution = false; 

//*************************************************************************************************
struct DVector
{
	double x, y, z;
};



short(*Mxyz)[DXYZ] = NULL;
double *HostMelo = NULL, *HostDens = NULL;
double Ts[DQ], Mass, ReTau, Tau, Viscosity, Radius, MaxSpeed, Width;
double Diameter, Gravity;
double DimLen, DimTime, DimMass, DimF, DimT, DimV, DimA, DimW, DimB;
double GravityX = 0, GravityY = 0, GravityZ = 0;
double K1, K2;
double SigmaI, SigmaL, deltaP, s, Laplace = 0;
double DenG, DenL, A, B, K, Ka, T, Tr, Tc, Rc, Pc, BasePt, BasePt2, CaLeft, CaCap=0, CaRight, DropVy, XLeft, XLeft_min= DX /2, XRight, XRight_max= DX / 2, DropMass=0,Mx= 0, My= 0, Mz= 0,My2,Mz2, XLeft0, XRight0, Hdown,starttime,endtime;//Hdown为液滴最低点高度
int    TasKNum = 1, No, NowStep, DropStep, AllStep, ShowStep, SaveStep, BeginTime, StepTime, LastTime, MModel, FModel;
int	   SpreadTime=0;


cudaError_t  err;
char    *Type;
double  *Dens, *Pote, *Dist, *Temp, *Vx, *Vy, *Vz, *MVx, *MVy, *MVz, *Fx, *Fy, *Fz;
double(*TDen)[DZ];
__constant__ double _K, _A, _B, _T, _Ka, _Kq, _DenG, _DenL, _Tau, _Ts[DQ], _ReTau, _NowStep, _DropStep, _DimA, _Gravity, _Radius, _Width,  _BasePt, _BasePt2, _K1, _K2;//_interfaceDen[DXY]

const int DR = 5;
const int LX = DX / DR, LY = DY / DR, LZ = DZ / DR;
const int LXYZ = LX*LY*LZ;

void Initialize();
void SetMultiphase();
void ShowData();
void CalcMacroCPU();
//void DataShowSave();




void SaveVel();//保存底板K=2液滴底面的速度
void SaveY_Vel();//流场中每一个X-Z平面的Y速度 
void SavePointVel();//保存每一时刻边缘亲水点的最大速度
void SaveAllByType();
void SaveSpreadtime();
void SaveDataOriDen();
void SaveContactAngle();
void Save_interfaceDen();//测试表面图案设置
void SaveData(double den,int a ,int b);
void DropletSplashSaveDrop();
void SaveTask();


int  GetMyTickCount();
int  TimeInterval();
void DeviceQuery();
void CudaInitialize();
void CudaFree();

//*************************************************************************************************
void ContactAngle(bool side, int tier);
void ComputeEnergy();
void Computetangential_velocity();
bool Cmp(const pair<double, double>& a, const pair<double, double>& b);
void DimConversion(double Length, double MacroLength, double Density, double MacroDensity, double Viscosity, double MacroViscosity);
double EosPressure(const double Density);
void GetThickness(const double Dens[], double & Width);
//double Sigma_IntegralMethod();

//*************************************************************************************************

dim3  DimBlock((DX*DY*DZ + 64 - 1) / 64); dim3 DimThread(64);
//dim3  DimBlock((DX*DY*DZ + 32 - 1) / 32); dim3 DimThread(32);
#define GridIndex   const int I = blockIdx.x*blockDim.x + threadIdx.x;
#define LineIndex   const int i = I / DY / DZ,  j = (I / DZ) % DY,  k = I % DZ;

//*************************************************************************************************
void Initialize()
{
	MModel = MP_CPPRW;
	Tau = 0.7; //Tau = 0.575;
	Tr = 0.68;
	K = 0.2;	//K = 0.1;	//计算三维液滴时取0.2;
	Ka = 0.001;
	K1 = 0; 
	K2 = -K1;
	//BasePt = 0.08;//-0.06(30.572°)   -0.04(49.094°)     0.02(91.975°)   0.072(120.107°)    0.082(124.678°)    0.112(137.039°)   0.132(143.795°)   0.15(149.187°)   0.18(160.378°)
	BasePt2 = -0.08;//-0.08;
	Radius = 60;
	Width = 10;
	Diameter = 0;// 0.05;//0.2;//cm//之前是0.025cm// +No*0.005;//cm //10微升液体半径大约0.25mm
	DropVy = -0.06;// -0.06;// -0.06;
	//TDen = new double[DY][DZ];//临时密度用于计算接触角 Y-Z平面
	TDen = new double[DX][DZ];//临时密度用于计算接触角 X-Z平面

	ReTau = double(1) / Tau;
	Viscosity = (Tau * 2 - 1) / 6;
	DimConversion(Radius * 2, Diameter, 7.2, 1.0, Viscosity, 0.01);
	Gravity = (Diameter == 0) ? 0 : -980 / DimA;
	//量纲变换,求出重力加速度;
	//cout << DimLen<<"	"<<DimTime <<"	"<< Gravity << endl;//Radius=30 Diameter=0.015 Tau=0.7 DimLen=0.00025 DimTime=4.16667e-7

	TasKNum = 1;
	ShowStep = 10000;
	SaveStep = 10000;
	AllStep = 30 * 10000;
	NowStep = StepTime = 0;
	DropStep = 500;//7000
	BeginTime = LastTime = GetMyTickCount();

	//fill((double*)TDen, (double*)TDen + DX * 4, 0);

	Ts[0] = Ts[3] = Ts[5] = Ts[7] = Ts[10] = Ts[12] = Ts[16] = Ts[17] = Ts[18] = 1.0;  //Ammar_2017_JCP;
	Ts[1] = 1;
    	Ts[2] = Ts[4] = Ts[6] = Ts[8] = 1.1;
	Ts[9] = Ts[11] = Ts[13] = Ts[14] = Ts[15] = 1.0 / Tau;  //*/

															/*Ts[0] = Ts[3] = Ts[5] = Ts[7] = 1.0;	Ts[1] = 1.19;
															Ts[2] = Ts[10] = Ts[12] = 1.4; 	Ts[4] = Ts[6] = Ts[8] = 1.2;
															Ts[16] = Ts[17] = Ts[18] = 1.98;
															Ts[9] = Ts[11] = Ts[13] = Ts[14] = Ts[15] = 1.0/Tau;  //*/
}

void SetParameter()
{
	//cout << "SetParameter" << endl;
	cudaMemcpyToSymbol(_K, &K, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(_A, &A, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(_B, &B, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(_T, &T, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(_Ka, &Ka, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(_K1, &K1, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(_K2, &K2, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(_DenG, &DenG, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(_DenL, &DenL, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(_Tau, &Tau, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(_ReTau, &ReTau, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(_BasePt, &BasePt, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(_BasePt2, &BasePt2, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(_Radius, &Radius, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(_Width, &Width, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(_Gravity, &Gravity, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(_Ts, Ts, sizeof(double)*DQ, 0, cudaMemcpyHostToDevice);
}


//*************************************************************************************************
__inline__ __device__ double Feq(int f, double Density, const double Vx, const double Vy, const double Vz)
{
	double DotMet = Vx*Ex[f] + Vy*Ey[f] + Vz*Ez[f];
	return Density * Alpha[f] * (1.0 + 3.0*DotMet + 4.5*DotMet*DotMet - 1.5*(Vx*Vx + Vy*Vy + Vz*Vz));
}




//*************************************************************************************************
__global__ void SetFlowField(char* Type, double* Dens, double* Pote, double* Dist, double* Temp )
{
		GridIndex;  LineIndex; if(I >= DXYZ) return;

#ifdef CONTACTANGLE
		if (k<3 || k>DZ - 4)
		{
			Type[I] = SOLID;
			Dens[I] = Pote[I] = _BasePt;
		}
		else 
		{
			Type[I] = FLUID;
			Dens[I] = Pote[I] = 0;
			DVector Vel;   Vel.x = Vel.y = Vel.z = 0;

			double r = sqrtf(Sq(D(i) - DX / 2) + Sq(D(j) - DY / 2) + Sq(D(k) - (_Radius + 2) )); // 液滴中心高度为Radius-20

			Dens[I] = (_DenL + _DenG) / 2 - (_DenL - _DenG) / 2 * tanh(D(r - _Radius) * 2 / _Width);
		}
#elif DROPLETSPLASH
		if (k<3 || k>DZ - 4)
		{
			Type[I] = SOLID;
			Dens[I] = Pote[I] = 0;
		}
		else 
		{
			Type[I] = FLUID;
			Dens[I] = Pote[I] = 0;
			DVector Vel;   Vel.x = Vel.y = Vel.z = 0;

			double Radius = 50, Width = 10, r = sqrtf(Sq(D(i) - DX / 2) + Sq(D(j) - DY / 2) + Sq(D(k) - DZ / 2));
			Dens[I] = _DenL +(_DenL-_DenG)/2 * (tanh(D(Radius-r)*2/Width) + tanh(D(32-k)*2/Width));		//液滴+液膜;
		}
#endif 

	for (int f = 0; f<DQ; ++f)
	{
		Dist[f*DXYZ + I] = Feq(f, Dens[I],0,0,0);
	}
	
}


//*************************************************************************************************
__device__ void LocalCollideMrt(double * Dens, double * Dist, const double Vx, const double Vy, const double Vz, int I, int i, int j, int k)
{
	double Den = Dens[I];
	double Meq[DQ], Mf[DQ], Mc[DQ];

	double Qp[19] = { 0 };//
		
	//Define_ijk5;
	//Qp[1] = -19 * (_K1 + 3 * _K2) * (GradX5(Dens)* GradX5(Dens) + GradY5(Dens) * GradY5(Dens) + GradZ5(Dens) * GradZ5(Dens));
	//Qp[9] = -_K1 * (2 * GradX5(Dens) * GradX5(Dens) - GradY5(Dens) * GradY5(Dens) - GradZ5(Dens) * GradZ5(Dens));
	//Qp[11] = -_K1 * (GradY5(Dens) * GradY5(Dens) - GradZ5(Dens) * GradZ5(Dens));
	//Qp[13] = -_K1 * GradX5(Dens) * GradY5(Dens);
	//Qp[14] = -_K1 * GradZ5(Dens) * GradY5(Dens);
	//Qp[15] = -_K1 * GradX5(Dens) * GradZ5(Dens);
	
	Define_ijk7;
	Qp[1] = -19 * (_K1 + 3 * _K2) * (Gradx7(Dens) * Gradx7(Dens) + Grady7(Dens) * Grady7(Dens) + Gradz7(Dens) * Gradz7(Dens));
	Qp[9] = -_K1 * (2 * Gradx7(Dens) * Gradx7(Dens) - Grady7(Dens) * Grady7(Dens) - Gradz7(Dens) * Gradz7(Dens));
	Qp[11] = -_K1 * (Grady7(Dens) * Grady7(Dens) - Gradz7(Dens) * Gradz7(Dens));
	Qp[13] = -_K1 * Gradx7(Dens) * Grady7(Dens);
	Qp[14] = -_K1 * Gradz7(Dens) * Grady7(Dens);
	Qp[15] = -_K1 * Gradx7(Dens) * Gradz7(Dens);

	Meq[0] = Den;
	Meq[1] = Den * (-11.0 + 19.0*(Vx*Vx + Vy*Vy + Vz*Vz));
	Meq[2] = Den * (3.0 - 11.0 / 2 * (Vx*Vx + Vy*Vy + Vz*Vz));
	Meq[3] = Den * Vx;
	Meq[4] = Den * Vx * -2 / 3;
	Meq[5] = Den * Vy;
	Meq[6] = Den * Vy * -2 / 3;
	Meq[7] = Den * Vz;
	Meq[8] = Den * Vz * -2 / 3;
	Meq[9] = Den * (Vx*Vx * 2 - Vy*Vy - Vz*Vz);
	Meq[10] = Den * (Vx*Vx * 2 - Vy*Vy - Vz*Vz) / -2;
	Meq[11] = Den * (Vy*Vy - Vz*Vz);
	Meq[12] = Den * (Vy*Vy - Vz*Vz) / -2;
	Meq[13] = Den * Vx * Vy;
	Meq[14] = Den * Vy * Vz;
	Meq[15] = Den * Vx * Vz;
	Meq[16] = 0;
	Meq[17] = 0;
	Meq[18] = 0;



	for (int f = 0; f<DQ; ++f)	//convert into the momentum space;
	{
		Mf[f] = 0;
		for (int i = 0; i < DQ; ++i)	Mf[f] += M[f][i] * Dist[DXYZ*i + I];
	}

	for (int f = 0; f<DQ; ++f)	//collide in the momentum sapce;
	{
		Mf[f] = Mf[f] - _Ts[f] * (Mf[f] - Meq[f]) + Qp[f] * _Ts[f]; // +Mc[f];
	}

	for (int f = 0; f<DQ; ++f)  //convert back to the lattice space;
	{
		Dist[DXYZ*f + I] = 0;
		for (int i = 0; i < DQ; ++i)  Dist[DXYZ*f + I] += R[f][i] * Mf[i];
	}
}   //*/


//*************************************************************************************************
#define DF(x)  Dist[(x)*DXYZ+I]
__global__ void GlobalCollide(char* Type, double* Dens, double* Pote, double* Dist, double* Temp, double *MVx, double *MVy, double *MVz, double *Vx, double *Vy, double*Vz, double *Fx, double * Fy, double *Fz)
{
	GridIndex;  LineIndex; if(I >= DXYZ) return;
	

	if (Type[I] == FLUID)
	{

		//计算宏观速度;
		MVx[I] = Vx[I] + Fx[I] / Dens[I] * 0.5;
		MVy[I] = Vy[I] + Fy[I] / Dens[I] * 0.5;
		MVz[I] = Vz[I] + Fz[I] / Dens[I] * 0.5;
		//double Melo = sqrt(Sq(MVx) + Sq(MVy) + Sq(MVz));   //宏观速度的模; //*/

		//计算平衡速度;
		double EVx = Vx[I] + Fx[I] / Dens[I];
		double EVy = Vy[I] + Fy[I] / Dens[I];
		double EVz = Vz[I] + Fz[I] / Dens[I];

		//多弛豫碰撞;
		LocalCollideMrt(Dens, Dist, Vx[I], Vy[I], Vz[I], I, i, j, k);

		for (int f = 0; f<DQ; ++f)
		{
			int ii = i + Ex[f];   if (ii < 0) ii += DX;  else if (ii >= DX) ii -= DX;
			int jj = j + Ey[f];   if (jj < 0) jj += DY;  else if (jj >= DY) jj -= DY;
			int kk = k + Ez[f];   if (kk < 0) kk += DZ;  else if (kk >= DZ) kk -= DZ;

			//Dist[f*DXYZ + I] -= 1./_Tau * (Dist[f*DXYZ + I] - Feq(f, Dens[I], Velo[I]));   //单弛豫模型;

			double Df = Dist[f*DXYZ + I] + Feq(f, Dens[I], EVx, EVy, EVz) - Feq(f, Dens[I], Vx[I], Vy[I], Vz[I]);  //精确差分力项;

			(Type[I(ii, jj, kk)] == FLUID ? Temp[f*DXYZ + I(ii, jj, kk)] : Temp[Re[f] * DXYZ + I]) = Df;  //流动和半程反弹;	

		}
	}
}


//*************************************************************************************************
__global__ void MacroCalculate(char* Type, double* Dens, double* Dist, double* Vx, double *Vy, double*Vz)
{
	GridIndex;  LineIndex; if(I >= DXYZ) return;

	if (Type[I] == FLUID)
	{
		Dens[I] = 0;
		for (int f = 0; f<DQ; ++f)
		{
			Dens[I] += Dist[f*DXYZ + I];
		}

		//计算格子速度;
		Vx[I] = (DF(1) + DF(7) + DF(9) + DF(11) + DF(13) - DF(2) - DF(8) - DF(10) - DF(12) - DF(14)) / Dens[I];
		Vy[I] = (DF(3) + DF(7) + DF(8) + DF(15) + DF(17) - DF(4) - DF(9) - DF(10) - DF(16) - DF(18)) / Dens[I];
		Vz[I] = (DF(5) + DF(11) + DF(12) + DF(15) + DF(16) - DF(6) - DF(13) - DF(14) - DF(17) - DF(18)) / Dens[I];
		//if (Dens[I] != Dens[I] || Dens[I] < 0 || Dens[I]>11)
		//{
		//printf("Density:  (%d,%d,%d)   %f     %f\n", i, j, k, Dens[I], Dens[I+1]);
		//}
	}
}


//*************************************************************************************************
__global__ void ChemBoundary(char * Type, double * Dens, double * Pote)
{
	const int i = blockIdx.x, j = threadIdx.x;   const int I = i*DY*DZ + j*DZ;
	const int i1 = (i>0 ? i - 1 : DX - 1), i2 = (i<DX - 1 ? i + 1 : 0), j1 = (j>0 ? j - 1 : DY - 1), j2 = (j<DY - 1 ? j + 1 : 0);

	int kk = 3;
	Dens[I] = Dens[I + 1] = Dens[I+2] = (Dens[I(i, j, kk)] * 2 + Dens[I(i1, j1, kk)] + Dens[I(i1, j2, kk)] + Dens[I(i2, j1, kk)] + Dens[I(i2, j2, kk)]) / 6;

	kk = DZ - 4;
	Dens[I+DZ-3] = Dens[I + DZ - 1] = Dens[I + DZ - 2] = (Dens[I(i, j, kk)] * 2 + Dens[I(i1, j1, kk)] + Dens[I(i1, j2, kk)] + Dens[I(i2, j1, kk)] + Dens[I(i2, j2, kk)]) / 6;

	Pote[I] = Pote[I + 1] = Pote[I + 2] = Pote[I + DZ -3] = Pote[I + DZ - 2] = Pote[I + DZ - 1] = _BasePt;

#ifdef COMPLEXCHEMBOUNDARY

	kk = 2;
	Dens[I + 1] = (Dens[I(i, j, kk)] * 2 + Dens[I(i1, j1, kk)] + Dens[I(i1, j2, kk)] + Dens[I(i2, j1, kk)] + Dens[I(i2, j2, kk)]) / 6;
	kk = DZ - 3;
	Dens[I + DZ - 2] = (Dens[I(i, j, kk)] * 2 + Dens[I(i1, j1, kk)] + Dens[I(i1, j2, kk)] + Dens[I(i2, j1, kk)] + Dens[I(i2, j2, kk)]) / 6;

	kk = 1;
	Dens[I] =  (Dens[I(i, j, kk)] * 2 + Dens[I(i1, j1, kk)] + Dens[I(i1, j2, kk)] + Dens[I(i2, j1, kk)] + Dens[I(i2, j2, kk)]) / 6;
	kk = DZ - 2;
	Dens[I + DZ - 1] = (Dens[I(i, j, kk)] * 2 + Dens[I(i1, j1, kk)] + Dens[I(i1, j2, kk)] + Dens[I(i2, j1, kk)] + Dens[I(i2, j2, kk)]) / 6;
#endif
}

__global__ void ChemBoundaryTag(char *Type , char originalPointType, char nextPointType)
{
	GridIndex;  LineIndex; if(I >= DXYZ) return;
    if (Type[I] == SOLID)
    {
        for (int f = 1; f < DQ; ++f)
        {
            if(i + Ex[f] >= DX || i + Ex[f] < 0 || j + Ey[f] >= DY || j + Ey[f] < 0 || k + Ez[f] >= DZ || k + Ez[f] < 0)  continue;

			int II = I(i + Ex[f], j + Ey[f], k + Ez[f]);
            if (Type[II] == nextPointType)
            {
					Type[I] = originalPointType;
					break;
            }
        }
    }
}

__global__ void ChemBoundaryComplex(char * Type, double * Dens, const char solid, const char fluid) 
{
	GridIndex;  LineIndex; if(I >= DXYZ) return;

    if(Type[I] == solid)
    {
        double avg_den = 0;
        double w = 0;

        for (int f = 1; f < DQ; ++f) 
        {
            const int xoffset = (i + Ex[f] + DX) % DX , yoffset = (j + Ey[f] + DY) % DY, zoffset = (k + Ez[f] + DZ) % DZ;
            const int pp = I(xoffset, yoffset, zoffset);

            if(Type [pp] == fluid)
            {
               avg_den += Alpha[f] * Dens[pp];
               w += Alpha[f];
            }
        }
        if(w) Dens[I] = avg_den / w;
		Pote[I] = _BasePt;
    }

}

__global__ void NonidealForce(char* Type, double* Dens, double* Pote, double* Fx, double* Fy, double* Fz)
{
	GridIndex;  LineIndex; if(I >= DXYZ) return;
	if (Type[I] == FLUID)
	{
		Define_ijk7;
		Fx[I] = -Gradx7(Pote) * Dens[I] + Gradx7(Dens) / 3;
		Fy[I] = -Grady7(Pote) * Dens[I] + Grady7(Dens) / 3;
		Fz[I] = -Gradz7(Pote) * Dens[I] + Gradz7(Dens) / 3;
	}
}

//*************************************************************************************************
__global__ void ChemPotential(char * Type, double * Dens, double * Pote)
{
	GridIndex;  LineIndex; if(I >= DXYZ) return;

	if (Type[I] == FLUID)
	{
		double Den = Dens[I], R0 = 9.7, W0 = 1.5;
		if (Den > R0) { Den = R0 + (Den - R0) / (D(1) + (Den - R0)*W0); }

		Pote[I] = _T*log(Den / (D(1) - _B*Den)) - _A / (S2 * 2 * _B)*log((S2 - 1 + _B*Den) / (S2 + 1 - _B*Den)) + _T / (D(1) - _B*Den) - _A*Den / (D(1) + _B*Den * 2 - Sq(_B*Den));

#ifdef PTCALCFIVEPOINT
		Define_ijk5;
		Pote[I] = Sq(_K) * Pote[I] - _Ka * GradD5(Dens);
#elif PTCALCSEVENPOINT
		Define_ijk7;
		Pote[I] = Sq(_K) * Pote[I] - _Ka * GradD7(Dens); 
#endif
	}
}

//*************************************************************************************************
__global__ void CompressField(double * Dens, double * Temp)
{
	const int i = blockIdx.x, j = blockIdx.y, k = threadIdx.x;
	const int I = i*LY*LZ + j*LZ + k;   if (I > LXYZ) return;
	const int II = (i*DY*DZ + j*DZ + k) * DR;
	Temp[I] = Dens[II];
}

void DimConversion(double Length, double MacroLength, double Density, double MacroDensity, double Viscosity, double MacroViscosity)
{
	DimLen = MacroLength / Length;
	DimTime = Viscosity / MacroViscosity * Sq(DimLen);
	DimMass = MacroDensity / Density * Sq(DimLen)*DimLen;

	DimF = DimMass * DimLen / Sq(DimTime);
	DimT = DimMass * Sq(DimLen) / Sq(DimTime);
	DimV = DimLen / DimTime;
	DimA = DimLen / Sq(DimTime);
	DimW = 1.0 / DimTime;
	DimB = 1.0 / Sq(DimTime);
}

//*************************************************************************************************
void DataShowSave()
{
	
	if (NowStep%ShowStep != 0 && NowStep%SaveStep != 0)   return;
	
	cudaMemcpy(HostDens, Dens, sizeof(double)*DXYZ, cudaMemcpyDeviceToHost);
	if ((err = cudaGetLastError()) != cudaSuccess)   cout << "CudaMemcpy: " << (int)err << "   " << cudaGetErrorString(err) << endl;

	Mass = 0;
	//for (int n = 0; n<DXYZ; ++n)
	//{
	//	int k = (n % (DY*DZ)) % DZ;
	//	if (k>1 && k<DZ - 2)
	//	{
	//		Mass += HostDens[n];
	//		if (MaxSpeed < HostMelo[n])   MaxSpeed = HostMelo[n];
	//	}
	//} //*/
	MaxSpeed = 0;
	for (int i = 0; i < DX; i++) {
		for (int j = 0; j < DY; j++) {
			for (int k = 2; k < DZ-2; k++) {
				//				cout << MVx[0] << endl;
				double vel = sqrt(Sq(MVx[I(i, j, k)]) + Sq(MVy[I(i, j, k)]) + Sq(MVz[I(i, j, k)]));
				//	if (k>1 && k<DZ-2)
				//		{
				Mass += Dens[I(i, j, k)];
				if (MaxSpeed < vel) MaxSpeed = vel;
				//	if (MaxSpeed < HostMelo[n])   MaxSpeed = HostMelo[n];
				//		}
			}
		}
	}

	if (NowStep > DropStep)
	{
		if (NowStep%SaveStep == 0)
		{
			SaveTask();
			//SaveAllByType(); 
			SaveVel();
		}
	}

	if (NowStep%ShowStep == 0)
	{
		TimeInterval();
		cout << "Laplace: "<<Laplace << endl;
		double We = DropVy * DropVy * 2 * DenL * Radius / Laplace;
		cout << NowStep << "   " << setiosflags(ios::fixed) << setprecision(12) << Mass << "   " << MaxSpeed << "   " << setprecision(3) << Tr << "   " << CaLeft << "   " << CaRight << "   " << CaCap << "   " << We << "   " << StepTime << endl;
	}

	/*if (NowStep%SaveStep == 0)
	{
	ofstream File;
	if( NowStep == SaveStep )
	{
	File.open( "data/CoExistence_PRW.txt" );
	File<<"DenG   Tr   DenL   Tr   Ratio"<<endl;
	File.close();
	}

	File.open( "CoExistence_PRW.txt", ios::app );
	File<<HostDens[I(DX/2,0,DZ/2)]/Rc<<"   "<<Tr<<"   "<< HostDens[I(DX/2,DY/2,DZ/2)]/Rc<<"   "<<Tr<<"   "<< HostDens[I(DX/2,DY/2,DZ/2)]/HostDens[I(DX/2,0,DZ/2)]<<endl;
	File.close();
	}  //*/
}

 void ContactAngle(bool side, int tier)
{
	double PI2 = 3.14159265358979323846264338327950288;
	double MidDen = (DenG + DenL) / 2;
	double Ca = 0;
	double Px[8] = { 0 };
	cudaMemcpy(HostDens, Dens, sizeof(double)*DXYZ, cudaMemcpyDeviceToHost);
	if (side == 1)
	{
		for (int j = 0; j <= DZ - 1; ++j)for (int i = 0; i < DY ; ++i)
		{
	
			TDen[i][j] = HostDens[I(tier, i, j)];//保存Y-Z平面上的密度
		}
	}
	else
	{
		for (int j = 0; j <= DZ - 1; ++j)  for (int i = 0; i<DX ; ++i)
		{
			TDen[i][j] = HostDens[I(i, tier, j)];//保存X-Z平面上的密度
		}
	}

	char FileName[256];
    	sprintf(FileName, "data/XZ_%f.txt" , BasePt);
	ofstream File(FileName);
	File << "i   j   Den" << endl;	
	for (int j = 0; j <= DZ - 1; ++j)  for (int i = 0; i<DX; ++i)
	{
		File << i << ' ' << j << ' ' << TDen[i][j] << endl;//保存X-Z平面上的密度
	}
	File.close();


	/*************************改进插值计算接触角*****************************/
	int begin = 2;//从j=3开始找界面点
				  //寻找液滴的界面

	for (int j = begin; j <= begin + 3; ++j)
	{
		int n = (j - begin) * 2;
		for (int i = 0; i < DX / 2 + 2; ++i)
		{			
			int ii = i + 1;
		int iii = i + 2;
			double &t0 = TDen[i][j], &t1 = TDen[ii][j], &t2 = TDen[iii][j];
			if (t1 < MidDen && t2 >= MidDen)
			{

				Px[n] = D(ii) + (t1 - MidDen) / (t1 - t2);	//第j行左边的气液界面位置;
				break;
			}
		}
		for (int i = DX - 2; i > DX / 2 - 2; --i)
		{
			int ii = i - 1;
			int iii = i - 2;
			double &t0 = TDen[i][j], &t1 = TDen[ii][j], &t2 = TDen[iii][j];
			if (t1 < MidDen && t2 >= MidDen)
			{

				//Px[n + 1] = D(iii) * ((MidDen - t0.Den) * (MidDen - t1.Den)) / ((t2.Den - t0.Den) * (t2.Den - t1.Den)) + D(ii) * ((MidDen - t0.Den) * (MidDen - t2.Den)) / ((t1.Den - t0.Den) * (t1.Den - t2.Den)) + D(i) * ((MidDen - t1.Den) * (MidDen - t2.Den)) / ((t0.Den - t1.Den) * (t0.Den - t2.Den));
				Px[n + 1] = D(ii) - (t1 - MidDen) / (t1 - t2);
				break;
			}
		}
	}
	//Px[0] Px[1]是固体点上的相界面

	//初始化值
	double y0 = 2;// 外推的高度
	double y1 = 3, y2 = 4, y3 = 5;
	//判断第三排是否有点
	if (Px[0] != D(0) && Px[1] != D(0))
	{
		//线性外推
		//左边接触点
		XLeft = Px[2] * ((y0 - y2) / (y1 - y2)) + Px[4] * ((y0 - y1) / (y2 - y1));
		//double x_L = Px[2] * ((y0 - y3) / (y2 - y3)) + Px[4] * ((y0 - y2) / (y3 - y2));
		//右边接触点
		XRight = Px[3] * ((y0 - y2) / (y1 - y2)) + Px[5] * ((y0 - y1) / (y2 - y1));
		// double x_R = Px[3] * ((y0 - y3) / (y2 - y3)) + Px[5] * ((y0 - y2) / (y3 - y2));
		 
		//接触角在175°以下用
		Ca = atan(D(y1 - y0) / (Px[2] - XLeft));
		if (Ca < 0)  Ca += PI2;   Ca = Ca / PI2 * 180;
		CaLeft = Ca;

		if(Ca > 175) cout << " ContactAngle > 175" << endl;

		Ca = atan(-D(y1 - y0) / (Px[3] - XRight));
		if (Ca < 0)  Ca += PI2;   Ca = Ca / PI2 * 180;
		CaRight = Ca;

		if(Ca > 175) cout << " ContactAngle > 175" << endl;
		//if (XLeft < XLeft_min)XLeft_min = XLeft;
		//if (XRight > XRight_max)XRight_max = XRight;

		////从计算液滴水平方向的中心点位置,计算液滴的最高点位置;
		double H, L = XRight - XLeft;

		//
		//for (int i = int((Px[4] + Px[5]) / 2 + 0.5), j = 2; j < DY - 2; ++j)//原 int i = int((Px[2] + Px[3]) / 2 + 0.5)
		for (int i = int((Px[4] + Px[5]) / 2 + 0.5), j = 2; j < DZ - 2; ++j)//原 int i = int((Px[2] + Px[3]) / 2 + 0.5)
		{
			double &t = TDen[i][j], &t1 = TDen[i][j + 1];
			if (t >= MidDen && t1 < MidDen)   H = D(j) + (t - MidDen) / (t - t1) - (y0);//-(y0+1) 因为外推的高度是第一层流体点
		}

		//当存在x_R - x_L为负数时，需要重新计算L，H
		if (L <= 0)
		{
			L = Px[3] - Px[2];
			H = H - 0.5;
		}
		if (XLeft>DY / 2 && XRight<DY / 2)
		{
			XLeft = DY / 2;
			XRight = DY / 2;
		}
		//采用球冠法用公式计算接触角;
		double R = (H * H * 4 + L * L) / H / 8;
		CaCap = atan(L / (R - H) / 2);
		if (CaCap < 0)  CaCap += PI2;   CaCap = CaCap / PI2 * 180;
	}
	else//第三排无值，角度为0，三相接触点在中间
	{
		XLeft =  DY / 2;
		XRight = DY / 2;
		CaLeft = 0;
		CaRight = 0;
	}

	// if (SpreadTime == 0 && XLeft <DY / 2 && XRight > DY / 2)
	// {
	// 	starttime = NowStep;
	// 	SpreadTime = 1;
	// }
	// if (SpreadTime == 1 && CaLeft==0 && CaRight==0)
	// {
	// 	endtime = NowStep + 1;
	// 	SaveSpreadtime();
	// 	SpreadTime = 2;
	// }


	// //找出液滴的质心  Y-Z平面
	// Mx = DX / 2;
	// My = 0;
	// My2 = 0;
	// Mz = 0;
	// Mz2 = 0;
	// Mass = 0;
	// DropMass = 0;
	// for (int j = 0; j <= DZ - 1; ++j)for (int i = 0; i < DY ; ++i)//积分法求质心位置  所有流体点
	// {
	// 	double &t = TDen[i][j];
	// 	Mass += t;
	// 	My += t*i;
	// 	Mz += t*j;

	// }
	// My /= Mass;
	// Mz /= Mass;
	// for (int j = 0; j <= DZ - 1; ++j)for (int i = 0; i < DY; ++i)//积分法求质心位置  所有流体点
	// {
	// 	double &t = TDen[i][j];
	// 	if (t >= (DenG + DenL) / 2)
	// 	{
	// 		DropMass += t;
	// 		My2 += t*i;
	// 		Mz2 += t*j;
	// 	}

	// }
	// My2 /= DropMass;
	// Mz2 /= DropMass;	
}

//计算液滴各个点速度
 void ComputeEnergy()
 {

	 ofstream File;
	 char FileName[256];
	 sprintf(FileName, "data/N4_Energy_or MV_%.3f.txt",BasePt);
	 if (NowStep == 1)
	 {
		 File.open(FileName);
		 File << "NowStep   Radius   Diameter   Tau   DropVy   V   VXY   VZ   MV   MVXY   MVZ   energy   energyXY   energyZ   Time" << endl;
		 File.close();
	 }	 
	 double V = 0;//整体液滴的速度
	 double VXY = 0;//液滴XY方向速度
	 double VZ = 0;//整体液滴的Z方向速度

	 double MV = 0;//整体液滴的动量 
	 double MVXY = 0;//液滴XY方向动量
	 double MVZ = 0;//液滴在Z方向动量

	 double energy = 0;//整体液滴的能量 
	 double energyXY = 0;//液滴XY方向能量
	 double energyZ = 0;//液滴在Z方向能量
	 
	 for (int i = 2; i < DX - 2; i++)
		 for (int j = 2; j < DY - 2; j++)
			 for (int k = 2; k < DZ - 2; k++)
			 {
				 if (Type[I(i, j, k)] == FLUID && Dens[I(i, j, k)]>(DenG+DenL)/2)
				 {
					 V   += sqrt(Sq(MVx[I(i, j, k)]) + Sq(MVy[I(i, j, k)]) + Sq(MVz[I(i, j, k)]));
					 VXY += sqrt(Sq(MVx[I(i, j, k)]) + Sq(MVy[I(i, j, k)]));
					 VZ  += MVz[I(i, j, k)];

					 MV    += Dens[I(i, j, k)] * sqrt(Sq(MVx[I(i, j, k)]) + Sq(MVy[I(i, j, k)]) + Sq(MVz[I(i, j, k)]));
					 MVXY  += Dens[I(i, j, k)] * sqrt(Sq(MVx[I(i, j, k)]) + Sq(MVy[I(i, j, k)]));
					 MVZ   += Dens[I(i, j, k)] * MVz[I(i, j, k)];

					 energy   += Dens[I(i, j, k)] * (Sq(MVx[I(i, j, k)]) + Sq(MVy[I(i, j, k)]) + Sq(MVz[I(i, j, k)]))/2;
					 energyXY += Dens[I(i, j, k)] * (Sq(MVx[I(i, j, k)]) + Sq(MVy[I(i, j, k)])) / 2;
					 energyZ  += Dens[I(i, j, k)] * Sq(MVz[I(i, j, k)]) / 2;
				 }
			 }
	 File.open(FileName, ios::app);
	 File << NowStep << "   " << Radius << "   " << Diameter << "   " << Tau << "   " << DropVy << "   " << V << "   " << VXY << "   " << VZ << "   " << MV << "   " << MVXY << "   " << MVZ << "   " << energy << "   " << energyXY << "   " << energyZ << "   " << NowStep*DimTime << endl;
	 File.close();  //*/
 }

 //计算液滴各个点的切向速度
 void Computetangential_velocity()
 {

	 //double alpha = atan(40.11 / 90);//里面的值需要double
	 //if (alpha < 0)  alpha += 3.1415926;   alpha = alpha / 3.1415926 * 180; //必须将弧度转化成角度
	 //cout << alpha << endl;
	 //cout << cos(60.0 / 180 * 3.1415926) << endl;
	 double PI2= 3.14159265358979323846264338327950288;
	 double tangential_vel = 0;		//垂直速度
	 double Mtangential_vel = 0;		//垂直速度动量
	 double mmv = 0;
	 double alpha = 0;				//格子点与水平方向Vx的夹角
	 double beta = 0;				//合速度V与水平速度Vx的夹角
	 double gamma = 0;				//
	 double theta = 0;				//合速度V在垂线速度V⊥的夹角
	 double V = 0;					//合速度
	 double tempV = 0;

	 ofstream File;
	 char FileName[256];
	 sprintf(FileName, "data/0722_N4_tangential_velocity_%.3f.txt", BasePt);
	 if (NowStep == 1)
	 {
		 File.open(FileName);
		 File << "NowStep   Radius   Diameter   Tau   DropVy   tangential_vel   Mtangential_vel   mmv   Time" << endl;
		 File.close();
	 }
	

	 for (int i = 2; i < DX - 2; i++)
		 for (int j = 2; j < DY - 2; j++)
			 for (int k = 2; k < DZ - 2; k++)  //k < DZ - 2
			 {
				 if (Type[I(i, j, k)] == FLUID && Dens[I(i, j, k)]>(DenG + DenL) / 2)
				 {
					 //以前的角度方案
					 //V = sqrt(Sq(MVx[I(i, j, k)]) + Sq(MVy[I(i, j, k)]));			//	XY平面的XY合速度
					 ////cout <<"	V	"<< V << endl;
					 //if (i < DX / 2 && j < DY / 2)
					 //{						 
						// beta = acos(fabs(MVx[I(i, j, k)]) / V);
						// //cout << "1 	beta	" << beta <<"	i:"<<i << "	j:" <<j << "	k:" <<k<< " 	MVx[I(i, j, k)])	" << MVx[I(i, j, k)] << "	MVy[I(i, j, k)])	" << MVy[I(i, j, k)] << endl;
						// beta = beta / PI2 * 180;									//  弧度转化为角度
						//// cout << "	beta/ PI2 * 180	" << beta << endl;

						// alpha = atan(fabs(D(j - DY / 2) / (i - DX / 2)));
						// //cout << "	alpha	" << alpha << endl;
						// alpha = alpha / PI2 * 180;									//  弧度转化为角度
						//// cout << "	alpha/ PI2 * 180	" << alpha << endl;

						// gamma = fabs(beta + alpha) / 180 * PI2;					//  角度转化为弧度，方便后续计算，可直接用

						// tempV = V * cos(PI2 / 2 - gamma);
					 //}
					 //else
					 //{	
						// if (i == DX / 2)
						// {
						//	 tempV = fabs(MVx[I(i, j, k)]);							//处于圆中线位置切向速度等于Vx
						// }
						// else
						// {
						//	 beta = acos(fabs(MVx[I(i, j, k)]) / V);
						//	 //cout << "2 	beta	" << beta << "	MVx[I(i, j, k)])	" << MVx[I(i, j, k)] << "	MVy[I(i, j, k)])	" << MVy[I(i, j, k)] << endl;
						//	 beta = beta / PI2 * 180;								//  弧度转化为角度
						//	// cout << "2	beta/ PI2 * 180	" << beta << endl;
						//	 alpha = atan(fabs(D(j - DY / 2) / (i - DX / 2)));
						//	 //cout << "2	alpha	" << alpha << endl;
						//	 alpha = alpha / PI2 * 180;								//  弧度转化为角度
						//	 //cout << "2	alpha/ PI2 * 180	" << alpha << endl;

						//	 gamma = fabs(beta - alpha) / 180 * PI2;				//  角度转化为弧度，方便后续计算，可直接用

						//	 tempV = V * cos(PI2 / 2 - gamma);
						// }
					 //}
					 //tangential_vel += tempV;
					 //Mtangential_vel += tempV*Dens[I(i, j, k)];
					 //mmv += tempV*tempV*Dens[I(i, j, k)] / 2;
					// cout <<i<<"	"<<j<< "	" << k <<  "	" << MVx[I(i, j, k)] << "	" << MVy[I(i, j, k)] << "	alpha:	"<< alpha<<"	beta:	"<< beta<<"		tempV:	" << tempV << "		" << endl;
					
					 //坐标变换方案
					 alpha = atan2(j - DY / 2, i - DX / 2) / PI2 * 180;
					 if (alpha < 0)alpha = alpha + 360; alpha = alpha / 180 * PI2;
					 //alpha = atan(fabs(D(j - DY / 2) / (i - DX / 2)));				 
					 //alpha = alpha / PI2 * 180;	//弧度转角度
					 //tangential_vel = MVx[I(i, j, k)] * sin(alpha) + MVy[I(i, j, k)] * cos(alpha);

					 //if (i >= DX / 2 && j >= DY / 2)c
					 //{
						// tempV = MVx[I(i, j, k)] * sin(alpha) + MVy[I(i, j, k)] * cos(alpha);
					 //}
					 //if (i < DX / 2 && j > DY / 2)
					 //{
						// tempV = MVx[I(i, j, k)] * sin(alpha) + MVy[I(i, j, k)] * cos(alpha);//PI2-alpha
					 //}
					 //if (i < DX / 2 && j < DY / 2)
					 //{
						// tempV = MVx[I(i, j, k)] * sin(alpha) + MVy[I(i, j, k)] * cos(alpha);//PI2 + alpha
					 //}
					 //if (i > DX / 2 && j < DY / 2)
					 //{
						// tempV = MVx[I(i, j, k)] * sin(alpha) + MVy[I(i, j, k)] * cos(alpha); //2 * PI2 - alpha
					 //}
					 tempV = MVx[I(i, j, k)] * sin(alpha) + MVy[I(i, j, k)] * cos(alpha);
					 tangential_vel += tempV;
					 Mtangential_vel += tempV * Dens[I(i, j, k)];
				 }
			 }
	 File.open(FileName, ios::app);
	 File << NowStep << "   " << Radius << "   " << Diameter << "   " << Tau << "   " << DropVy << "   " << tangential_vel << "   " << Mtangential_vel << "   " << mmv << "   " << NowStep*DimTime << endl;
	 File.close();  //*/
 }


void SaveTask()
{
	dim3  Block(LX, LY, 1), Thread(LZ, 1, 1);
	CompressField << < Block, Thread >> > (Dens, Temp);
	cudaMemcpy(HostDens, Temp, sizeof(double)*LXYZ, cudaMemcpyDeviceToHost);//压缩用这三行
	//cudaMemcpy(HostDens, Dens, sizeof(double)*DXYZ, cudaMemcpyDeviceToHost);//不压缩用这

	char FileName[256];
	sprintf(FileName, "FlowField_%d", NowStep);
	//sprintf(FileName, "FlowField_%d_Pt", int(BasePt * 100));

	TEC_FILE tec_file(FileName);
	tec_file.Title = "Model3D";
	tec_file.Variables.push_back("i");
	tec_file.Variables.push_back("j");
	tec_file.Variables.push_back("k");
	tec_file.Variables.push_back("Density");
	tec_file.Zones.push_back(TEC_ZONE(FileName));
	tec_file.Zones[0].Max[0] = LZ;//保存所有DZ    压缩保存为LZ
	tec_file.Zones[0].Max[1] = LY;//保存所有DY    压缩保存为LY
	tec_file.Zones[0].Max[2] = LX;//保存所有DX    压缩保存为LX
	tec_file.Zones[0].Data.push_back(TEC_DATA(Mxyz[0]));
	tec_file.Zones[0].Data.push_back(TEC_DATA(Mxyz[1]));
	tec_file.Zones[0].Data.push_back(TEC_DATA(Mxyz[2]));
	tec_file.Zones[0].Data.push_back(TEC_DATA(HostDens));
	tec_file.write_plt(1);
}

void DropletSplashSaveDrop()
{
	if (NowStep == DropStep)
	{
		for(int i = 2;i < DX - 2;i++)
			for (int j = 2; j < DY - 2; j++)
				for (int k = 3; k < DZ - 3; k++)
				{
					if (Type[I(i, j, k)] == FLUID)
					{
						double r = sqrtf(Sq(D(i) - DX / 2) + Sq(D(j) - DY / 2) + Sq(D(k) - DZ/2));
						if (r <= 50 && Dens[I(i,j,k)]>(DenG + DenL) / 2) {
							
							//Vz[I(i,j,k)] += DropVy;
							//cout << i << "\t" << j << "\t" << k << "\t" << Vz[I(i, j, k)] << endl;
							//cin.get();
						}
					}
				}
	}

	//if (NowStep == 2000 || NowStep == 200||NowStep == 3500 ||NowStep == 4000 ||NowStep == 4700)
	//{
	//	//cout <<NowStep << endl;
	//	SaveDataOriDen();
	//}

	ofstream File;
	char FileName[256];
	sprintf(FileName, "data/Spread_Drop_%.3f.txt", BasePt);
	if (NowStep == 1)
	{		
		File.open(FileName);
		File << "Droprad   BasePt   NowStep   NowStep*DimTime   CaLeft   CaRight   XLeft   XRight   L   Mx   My   Mz   My2   Mz2" << endl;
		File.close();
	}

	File.open(FileName, ios::app);
	File << Radius << "   " << BasePt << "   " << NowStep << "   " << NowStep*DimTime << "   " << CaLeft << "   " << CaRight << "   " << XLeft << "   " << XRight << "   " << (XRight-XLeft)*DimLen << "   " << Mx << "   " << My*DimLen << "   " << Mz*DimLen << "   " << My2*DimLen << "   " << Mz2*DimLen << endl;
	File.close();  //*/

	//ofstream File;
	//if (No == 0)
	//{
	//	File.open("data/Bounce_Drop.txt");
	//	File << "Droprad   BasePt   CaLeft   CaRight   XLeft_min   XRight_max   L" << endl;
	//	File.close();
	//}

	//File.open("Bounce_Drop.txt", ios::app);
	//File << Radius << "   " << BasePt << "   " << CaLeft << "   " << CaRight << "   " << XLeft_min << "   " << XRight_max << "   " << XRight_max - XLeft_min << endl;
	//File.close();  //*/
	//XLeft_min = DX/2;
	//XRight_max = DX/2;
}
//流场中每一个点的动量 //速度以及密度
void SaveVel()
{
	ofstream File;
	char FileName[256];
	sprintf(FileName, "data/N=4_MVel_%dBasePt_%.3f.txt", NowStep,BasePt);	
	if (No == 0)
	{ 
		File.open(FileName);
		File << "i   j   MV" << endl;
		File.close();
	}
	
	for (int i = 0; i < DX; ++i)
		for (int j = 0; j < DY; ++j)
			//if(Dens[I(i,j,2)]>=(DenG + DenL) / 2)
			{
				File.open(FileName, ios::app);
				File << i << "   " << j << "   " << (Dens[I(i, j, 2)] >= (DenG + DenL) / 2 ? Dens[I(i, j, 2)] * sqrt(MVx[I(i, j, 2)] * MVx[I(i, j, 2)] + MVy[I(i, j, 2)] * MVy[I(i, j, 2)] + MVz[I(i, j, 2)] * MVz[I(i, j, 2)]):0) << endl;//保存动量    + MVz[I(i, j, 2)] * MVz[I(i, j, 2)]
				//File <<i << "   " << j << "   " << sqrt(MVx[I(i, j, 2)] * MVx[I(i, j, 2)] + MVy[I(i, j, 2)] * MVy[I(i, j, 2)] + MVz[I(i, j, 2)] * MVz[I(i, j, 2)]) << endl;//保存速度
				File.close();
			}
}

//流场中每一个X-Z平面的Y速度 //速度以及密度
void SaveY_Vel()
{
	ofstream File;
	char FileName[256];
	sprintf(FileName, "data/N=4_Yvel_XZ_%dBasePt_%.3f.dat", NowStep, BasePt);
	if (No == 0)
	{
		File.open(FileName);
		File << "TITLE = Droplet" << endl;
		File << "VARIABLES =  X,   Y,   Z,   Uy" << endl;
		File << "ZONE  I=" << DX << "  Y=" << DY << "  Z=" << DZ << "  F=POINT " << endl;
		File.close();
	}

	FOR_iDX_jDY_kDZ
	{
		File.open(FileName, ios::app);
		File << i << "   " << j << "   " << k << "   " << (Dens[I(i, j, k)] >= (DenG + DenL) / 2 ? MVy[I(i, j, k)]*DimV :-2)<< endl;//保存动量    + MVz[I(i, j, 2)] * MVz[I(i, j, 2)]
																																			   //File <<i << "   " << j << "   " << sqrt(MVx[I(i, j, 2)] * MVx[I(i, j, 2)] + MVy[I(i, j, 2)] * MVy[I(i, j, 2)] + MVz[I(i, j, 2)] * MVz[I(i, j, 2)]) << endl;//保存速度
		File.close();
	}
}

void SaveAllByType()
{
	//cudaMemcpy(Type, _Type, sizeof(int)*DXYZ, cudaMemcpyDeviceToHost);

	int count = DXYZ;
	double *ai = new double[count];
	double *aj = new double[count];
	double *ak = new double[count];
	double *aden = new double[count];
	double *Ux = new double[count];
	double *Uy = new double[count];
	double *Uz = new double[count];
	for (int i = 0; i < DX; ++i)
	{
		for (int j = 0; j < DY; ++j)
		{
			for (int k = 0; k < DZ; ++k)
			{
				int p = i*DY*DZ + j*DZ + k;
				int xyz = i + j*DX + k*DY*DX;
				ai[xyz] = i;
				aj[xyz] = j;
				ak[xyz] = k;
				aden[xyz] = Dens[I(i, j, k)];
				Ux[xyz] = (Dens[I(i, j, k)] >= (DenG + DenL) / 2 ? MVx[I(i, j, k)] * DimV : -2);				
				Uy[xyz] = (Dens[I(i, j, k)] >= (DenG + DenL) / 2 ? MVy[I(i, j, k)] * DimV : -2);
				Uz[xyz]= (Dens[I(i, j, k)] >= (DenG + DenL) / 2 ? MVz[I(i, j, k)] * DimV : -2);
			}
		}
	}
	char FileName[128];
	sprintf(FileName, "Field_3D_Nowstep=%d", NowStep);
	TEC_FILE tec_file(FileName);
	tec_file.Title = "Field_3D";
	tec_file.Variables.push_back("i");
	tec_file.Variables.push_back("j");
	tec_file.Variables.push_back("k");
	tec_file.Variables.push_back("Ux");
	tec_file.Variables.push_back("Uy");
	tec_file.Variables.push_back("Uz");
	tec_file.Variables.push_back("Density");
	tec_file.Zones.push_back(TEC_ZONE(FileName));
	tec_file.Zones[0].Max[0] = DX;
	tec_file.Zones[0].Max[1] = DY;
	tec_file.Zones[0].Max[2] = DZ;
	tec_file.Zones[0].Data.push_back(TEC_DATA(ai));
	tec_file.Zones[0].Data.push_back(TEC_DATA(aj));
	tec_file.Zones[0].Data.push_back(TEC_DATA(ak));
	tec_file.Zones[0].Data.push_back(TEC_DATA(Ux));
	tec_file.Zones[0].Data.push_back(TEC_DATA(Uy));
	tec_file.Zones[0].Data.push_back(TEC_DATA(Uz));
	tec_file.Zones[0].Data.push_back(TEC_DATA(aden));
	tec_file.write_plt(1);
	delete[] ai;
	delete[] aj;
	delete[] ak;
	delete[] Ux;
	delete[] Uy;
	delete[] Uz;
	delete[] aden;
}


void SavePointVel()
{
	ofstream File;
	char FileName[256];
	sprintf(FileName, "data/PointVel_Diameter%f_BasePt%.3f.txt", Diameter,BasePt);
	if (NowStep == 1)
	{
		File.open(FileName);
		File << "Droprad   NowStep   NowStep*DimTime   Pote   i   j   k   Dens   MVx   MVy   MVz   MVmax" << endl;
		File.close();
	}

	//File.open(FileName, ios::app);
	//cout << 111 << endl;
	for (int i = 0; i < DX; i++)
		for (int j = 0; j < DY; j++)
			
			if (Dens[I(i, j, 2)] >= (DenG + DenL) / 2 && Pote[I(i, j, 1)] == BasePt2 && Dens[I(i+1, j, 2)] < (DenG + DenL) / 2 && Dens[I(i, j+1, 2)] < (DenG + DenL) / 2)    // && Pote[I(i, j, 2)] == BasePt2   
			{
				//cout << 222 << i << "   " << j << "   " << 2 << endl;
				File.open(FileName, ios::app);
				File << Radius << "   " << NowStep << "   " << NowStep*DimTime << "   " << Pote[I(i, j, 1)] << "   " << i << "   " << j << "   " << 2 << "   " << Dens[I(i, j, 2)] << "   " << MVx[I(i, j, 2)] << "   " << MVy[I(i, j, 2)] << "   " << MVz[I(i, j, 2)] << "   " << MaxSpeed << endl;
				break;
				
			}
	File.close();

}

void SaveSpreadtime()
{

	ofstream File;
	if (No == 0)
	{
		File.open("data/time.txt");
		File << "Droprad   Diameter   BasePt   Tau   DropVy   startTime   endtime   spreadtime   vel" << endl;
		File.close();
	}

	double mvel = 0;
	double vel = 0;
	double dropmass = 0;
	for (int i = 0; i < DX; i++) 
	{
		for (int j = 0; j < DY; j++) 
		{
			for (int k = 2; k < DZ - 2; k++)
			{
				if (Dens[I(i, j, k)] > (DenG + DenL) / 2)
				{
					mvel += Dens[I(i, j, k)]* sqrt(Sq(MVx[I(i, j, k)]) + Sq(MVy[I(i, j, k)]) + Sq(MVz[I(i, j, k)]));//累加求出液滴的动量
					dropmass += Dens[I(i, j, k)];//累加求出液滴的质量
				}
			}
		}
	}

	vel = mvel / dropmass;//液滴离开瞬间总体的速度
	File.open("time.txt", ios::app);
	File << Radius << "   " <<Diameter << "   " << BasePt << "   " <<Tau<< "   " << DropVy << "   " << starttime << "   " << endtime << "   " << (endtime-starttime)*DimTime << "   " << vel << endl;
	File.close();  //*/
}

void SaveDataOriDen()
{
	//计算表面张力,分别采用拉普拉斯定律和积分法计算;
	double MidPos = DY / 2 - Radius, MidDen = (Dens[I(DX / 2, 0, DZ / 2)] + Dens[I(DX / 2, DY / 2, DZ / 2)]) / 2;
	double TheRad = 0, Integral = 0;
	for (int j = 0; j <= DY / 2; ++j)
	{
		double &t1 = Dens[I(DX/2, j , DZ / 2)], &t2 = Dens[I(DX / 2, j+1, DZ / 2)];
		if (t1 <= MidDen && t2>MidDen)
		{
			MidPos = D(j) + (MidDen - t1) / (t2 - t1);
			TheRad = D(DY / 2) - MidPos;
			break;
		}
	}


	double &t1 = Dens[I(DX / 2, 0, DZ / 2)], &t2 = Dens[I(DX / 2, DY / 2, DZ / 2)];
	//cout << t1 << " " << t2 << " " << endl;
	double Pr0_In = EosPressure(t2);
	double Pr0_Out = EosPressure(t1);
	//cout << Pr0_In << "	" << Pr0_Out << endl;
	Laplace = (K) * (Pr0_In - Pr0_Out) * TheRad / 2;  //拉普拉斯定律求表面张力;


	char TrackName[256];
	ofstream File;
	sprintf(TrackName, "data/3Lacplace_%s_Tau%3.2f_Tr%3.2f.txt", Name(MModel), Tau, Tr);
	if (NowStep == 0)//No==0 NowStep == 501
	{
		File.open(TrackName);
		File << "Tr   Radius   TheRad   1/Rad   DeltaP   Laplace" << endl;
	}
	else
	{
		File.open(TrackName, ios::app);
	}

	File << Tr << "   " << Radius << "   " << TheRad << "   " << 1.0 / TheRad << "   " << Pr0_In - Pr0_Out << "   " << Laplace <<  endl;
	File.close();
}

void Save_interfaceDen()
{	
	//看底部图形origin
	cudaMemcpy(HostDens, Dens, sizeof(double)*DXYZ, cudaMemcpyDeviceToHost);
	ofstream File;
	if (No == 0)
	{

		File.open("data/interface_Den.txt");
		File << "i   j" << endl;
		File.close();
	}
	for (int i = 0; i < DX - 1; i++)
		for (int j = 0; j < DY - 1; j++)
		{
			if (HostDens[I(i, j, 1)] == -2)
			{
				File.open("interface_Den.txt", ios::app);
				File << i << "   " << j << endl;//<< "   " << HostDens[I(i, j, 1)] 
				File.close();
			}
			
		}
	
	//cudaMemcpy(HostDens, Dens, sizeof(double)*DXYZ, cudaMemcpyDeviceToHost);
	//char FileName[256];
	//ofstream File;
	//sprintf(FileName, "data/Try_Compare_FlowField_%d.dat", NowStep);
	//File.open(FileName);
	//File << "TITLE = Droplet" << endl;
	//File << "VARIABLES =  X,   Y,   Z,   Density" << endl;
	//File << "ZONE  I=" << DX << "  J=" << DY << "  K=" << DZ << "  F=POINT " << endl;
	////int z = 1;
	//for (int i = 0; i < DX; i++)
	//	for (int j = 0; j < DY ; j++)
	//		for (int k = 0; k < DZ; k++)
	//		{
	//			File << i << ' ' << j << ' ' << k << ' ' << setprecision(6) << HostDens[I(i, j, k)] << endl;
	//			
	//		}
		
}

//*************************************************************************************************
int main(int argc, char *argv[])
{
	DeviceQuery();
	int DeviceNo = *argv[1] - '0';
	// cout << " Which GPU device do you want to use? ";  
	// cin.get(&DeviceNo); 
	cudaSetDevice(DeviceNo); //cudaSetDevice(1);
	cout << "  Now is running on GPU device " << DeviceNo << endl;

	No = 0;
	//for( BasePt = 0.01 ;BasePt >= -0.06; BasePt -= 0.005)
	//for (; No < TasKNum; ++No)
	{

		Initialize();
		SetMultiphase();
		CudaInitialize();
		SetFlowField << <DimBlock, DimThread >> > (Type, Dens, Pote, Dist, Temp);

		//标记化学势边界点
		ChemBoundaryTag << <DimBlock, DimThread >> > (Type, LEVEL1, FLUID); // (type, originPointType, nextPointType)
		ChemBoundaryTag << <DimBlock, DimThread >> > (Type, LEVEL2, LEVEL1);
		ChemBoundaryTag << <DimBlock, DimThread >> > (Type, LEVEL3, LEVEL2);

		for( BasePt = -0.08 ;BasePt <=0.14; BasePt += 0.005, No++)
		{
			cout << "Now the BasePt = " << BasePt << endl;

			cudaMemcpyToSymbol(_BasePt, &BasePt, sizeof(double));
			cudaDeviceSynchronize();
			NowStep = 0;

			CalcMacroCPU();
			ShowData();

			for (NowStep = 1; NowStep <= AllStep; ++NowStep)
			{
				dim3  Block(DX, 1, 1), Thread(DY, 1, 1);
				{
					//ChemBoundary<< <Block, Thread >> > (Type, Dens, Pote);

					ChemBoundaryComplex<< <DimBlock, DimThread >> > (Type, Dens, LEVEL1, FLUID);
					ChemBoundaryComplex<< <DimBlock, DimThread >> > (Type, Dens, LEVEL2, LEVEL1);
					ChemBoundaryComplex<< <DimBlock, DimThread >> > (Type, Dens, LEVEL3, LEVEL2);
				
					ChemPotential << <DimBlock, DimThread >> > (Type, Dens, Pote);
				}

				NonidealForce << <DimBlock, DimThread >> > (Type, Dens, Pote, Fx, Fy, Fz);
				GlobalCollide << <DimBlock, DimThread >> > (Type, Dens, Pote, Dist, Temp, MVx, MVy, MVz, Vx, Vy, Vz, Fx, Fy, Fz);
				double *p = Dist;  Dist = Temp;  Temp = p;
				MacroCalculate << <DimBlock, DimThread >> > (Type, Dens, Dist, Vx, Vy, Vz);
				cudaDeviceSynchronize();

				// DropletSplashSaveDrop();//存每时步液滴扩散半径、质心位置以及接触角
	
				if (NowStep%ShowStep == 0 && NowStep%SaveStep == 0)   
				{
					CalcMacroCPU();
					ContactAngle(0, DY / 2); //ContactAngle(1, DX / 2);
					ShowData();
				}
				if( NowStep == AllStep) SaveContactAngle();
			}
		}
		CudaFree();
	}

	cout << endl << " Press Enter key to quit ...... ";   cin.get();
	delete[] TDen;
	return 0;
}

//*************************************************************************************************
void SetMultiphase()
{
	Viscosity = (Tau * 2 - 1) / 6;

	switch (MModel)
	{
	case MP_SCVDW: case MP_CPVDW: case MP_P0VDW:
		A = D(9) / 49;   B = D(2) / 21;   Tc = D(4) / 7;     Rc = D(7) / 2;
		break;

	case MP_SCCSE: case MP_CPCSE: case MP_P0CSE:   //注意此处B为除4后的值;
		A = 1.0;       B = 1.0;   Tc = 0.09432870;     Rc = 0.13044388;
		break;

	case MP_SCRKE: case MP_CPRKE: case MP_P0RKE:   //注意此处A为合并alpha系数后的值;
		A = D(2) / 49;   B = D(2) / 21;   Tc = 0.196133;   Rc = 2.729171;
		A *= D(1) / sqrt(Tc*Tr);
		break;

	case MP_SCRKS: case MP_CPRKS: case MP_P0RKS: { //注意此处A为合并alpha系数后的值;
		A = D(2) / 49;   B = D(2) / 21;   Tc = 0.086861;   Rc = 2.729171;  double w = 0.344;
		A *= Sq(D(1) + (0.480 + 1.574*w - 0.176*w*w)*(D(1) - sqrt(Tr)));
	}break;

	case MP_SCPRM: case MP_CPPRM: case MP_P0PRM:
	case MP_SCPRW: case MP_CPPRW: case MP_P0PRW: { //注意此处A为合并alpha系数后的值;
		A = D(2) / 49;  B = D(2) / 21;  Tc = 0.072919;  Rc = 2.65730416;
		double w = (MModel == MP_SCPRW || MModel == MP_CPPRW || MModel == MP_P0PRW) ? 0.344 : 0.011;
		A *= Sq(D(1) + (0.37464 + 1.54226*w - 0.26992*w*w)*(D(1) - sqrt(Tr)));
	}break;
	}

	//check the _Temperature;
	if (Tr < 0.1 && Tr>1)
	{
		cout << "_Temperature is error:  " << Tr << endl;
		return;
	}

	T = Tr * Tc;

	//Read the _Densities of gas and liquid phases;
	string FileName = Name(MModel);
	FileName = "CoCurve/CoCurve_" + FileName.substr(3, 3) + "_2.txt";

	ifstream File(FileName.c_str());
	if (!File.is_open())
	{
		cout << "open file error:  " << FileName << endl;
		return;
	}

	char Buffer[512];
	istringstream Iss;
	DenG = DenL = 0;

	while (!File.eof())
	{
		double T0, DenGas, DenLiquid;
		File.getline(Buffer, 512);
		Iss.clear();  Iss.str(Buffer);
		Iss >> T0 >> DenGas >> DenLiquid;

		if (Eq(T0, Tr))
		{
			DenG = DenGas * Rc;
			DenL = DenLiquid * Rc;
			break;
		}
	}
	File.close();

	if (NowStep == 0)//if (NowStep == 0)
	{
		//SigmaI = Sigma_IntegralMethod();
		double Re = abs(DropVy) * 2 * Radius * 6 / (0.55 * 2 - 1);
		double We = DropVy * DropVy * 2 * DenL * Radius  / Laplace;
		cout << endl << " Multiphase: " << Name(MModel) << "  (" << DX << ", " << DY << ", " << DZ << ")   Tau=" << setprecision(2) << Tau << "   Tr=" << Tr << "   DenL=" << setprecision(5) << DenL << "   Re=" << Re << "   We=" << We << endl;
		cout << "****************************************************************************************************" << endl;
	}
}


//*************************************************************************************************
void CudaInitialize()
{
	cudaMallocManaged((void**)&Type, sizeof(char)   *DX*DY*DZ);
	cudaMallocManaged((void**)&Dens, sizeof(double) *DX*DY*DZ);
	cudaMallocManaged((void**)&Pote, sizeof(double) *DX*DY*DZ);
	cudaMallocManaged((void**)&Dist, sizeof(double) *DX*DY*DZ*DQ);
	cudaMallocManaged((void**)&Temp, sizeof(double) *DX*DY*DZ*DQ);

	cudaMallocManaged((void**)&Fx, sizeof(double)  *DX*DY*DZ);
	cudaMallocManaged((void**)&Fy, sizeof(double)  *DX*DY*DZ);
	cudaMallocManaged((void**)&Fz, sizeof(double)  *DX*DY*DZ);
	cudaMallocManaged((void**)&Vx, sizeof(double)  *DX*DY*DZ);
	cudaMallocManaged((void**)&Vy, sizeof(double)  *DX*DY*DZ);
	cudaMallocManaged((void**)&Vz, sizeof(double)  *DX*DY*DZ);
	cudaMallocManaged((void**)&MVx, sizeof(double)  *DX*DY*DZ);
	cudaMallocManaged((void**)&MVy, sizeof(double)  *DX*DY*DZ);
	cudaMallocManaged((void**)&MVz, sizeof(double)  *DX*DY*DZ);

	if ((err = cudaGetLastError()) != cudaSuccess)   cout << "CudaMalloc : " << (int)err << "   " << cudaGetErrorString(err) << endl;

	cudaMemcpyToSymbol(_Ts, &Ts, sizeof(double)*DQ);
	cudaMemcpyToSymbol(_Tau, &Tau, sizeof(double));
	cudaMemcpyToSymbol(_K, &K, sizeof(double));
	cudaMemcpyToSymbol(_T, &T, sizeof(double));
	cudaMemcpyToSymbol(_A, &A, sizeof(double));
	cudaMemcpyToSymbol(_B, &B, sizeof(double));
	cudaMemcpyToSymbol(_Ka, &Ka, sizeof(double));
	cudaMemcpyToSymbol(_K1, &K1, sizeof(double));
	cudaMemcpyToSymbol(_K2, &K2, sizeof(double));
	cudaMemcpyToSymbol(_DenL, &DenL, sizeof(double));
	cudaMemcpyToSymbol(_DenG, &DenG, sizeof(double));
	cudaMemcpyToSymbol(_BasePt, &BasePt, sizeof(double));
	cudaMemcpyToSymbol(_BasePt2, &BasePt2, sizeof(double));
	//液滴半径 、直径、重力、Z方向重力
	cudaMemcpyToSymbol(_Radius, &Radius, sizeof(double));
	cudaMemcpyToSymbol(_Width, &Width, sizeof(double));
	cudaMemcpyToSymbol(_NowStep, &NowStep, sizeof(double));
	cudaMemcpyToSymbol(_DropStep, &DropStep, sizeof(double));
	cudaMemcpyToSymbol(_Gravity, &Gravity, sizeof(double));
	if ((err = cudaGetLastError()) != cudaSuccess)   cout << "CudaSymbol : " << (int)err << "   " << cudaGetErrorString(err) << endl;

	Mxyz = new short[3][DXYZ];
	HostDens = new double[DXYZ];
	HostMelo = new double[DXYZ];


	//压缩时写法
	for (int i = 0; i<LX; ++i)  for (int j = 0; j<LY; ++j)  for (int k = 0; k<LZ; ++k)
	{
		int n = i*LY*LZ + j*LZ + k;
		Mxyz[0][n] = i;
		Mxyz[1][n] = j;
		Mxyz[2][n] = k;
	}

	////不压缩时方法
	//for (int i = 0; i<DX; ++i)  for (int j = 0; j<DY; ++j)  for (int k = 0; k<DZ; ++k)
	//{
	//	int n = i*LY*LZ + j*LZ + k;	
	//	Mxyz[0][I(i, j, k)] = i;
	//	Mxyz[1][I(i, j, k)] = j;
	//	Mxyz[2][I(i, j, k)] = k;
	//}  //*/
}

void CudaFree()
{
	delete[] Mxyz;
	delete[] HostDens;
	delete[] HostMelo;

	cudaFree(Type);
	cudaFree(Dist);
	cudaFree(Temp);
	cudaFree(Dens);
	cudaFree(Pote);
	cudaFree(Vx);
	cudaFree(Vy);
	cudaFree(Vz);
	cudaFree(MVx);
	cudaFree(MVy);
	cudaFree(MVz);
	cudaFree(Fx);
	cudaFree(Fy);
	cudaFree(Fz);

	if ((err = cudaGetLastError()) != cudaSuccess)   cout << "CudaFree : " << (int)err << "   " << cudaGetErrorString(err) << endl;
}

//*************************************************************************************************
void DeviceQuery()
{
	//cudaDeviceReset();

	int deviceCount = 0, driverVersion = 0;
	err = cudaGetDeviceCount(&deviceCount);

	if (err != cudaSuccess)
	{
		printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)err, cudaGetErrorString(err));
		return;
	}

	if (deviceCount == 0)
	{
		printf("There is no available device that support CUDA\n");
	}
	else
	{
		printf("Detected %d CUDA Capable device(s)\n", deviceCount);
	}

	for (int n = 0; n < deviceCount; ++n)
	{
		cudaSetDevice(n);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, n);
		cudaDriverGetVersion(&driverVersion);

		printf("Device %d: %s    Ver:%d.%d/%d.%d    Core:%dx%d=%d    Memory:%.3f GB\n", n, \
			deviceProp.name, driverVersion / 1000, (driverVersion % 100) / 10, deviceProp.major, deviceProp.minor,
			deviceProp.multiProcessorCount, _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
			_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount,
			(float)deviceProp.totalGlobalMem / 1024 / 1024 / 1024);
	}

	cudaSetDevice(0);
	if ((err = cudaGetLastError()) != cudaSuccess)   cout << "DeviceQuery : " << (int)err << "   " << cudaGetErrorString(err) << endl;
}


//*************************************************************************************************
int GetMyTickCount()
{
#ifdef WIN32
	return ::GetTickCount();
#else
	struct timeval tv;	 gettimeofday(&tv, NULL);
	return (tv.tv_sec * 1000000 + tv.tv_usec) / 1000;
#endif
}
int  TimeInterval()
{
	int NowTime = GetMyTickCount();
	StepTime = NowTime - LastTime;
	LastTime = NowTime;
	return StepTime;
}

bool Cmp(const pair<double, double>& a, const pair<double, double>& b)
{
	return a.second < b.second;
}

void SaveData(double den,int i ,int j)
{
	ofstream File("data/Try_Compare.txt");
	File << "TDen   i   j" << endl;
	File << den << ' ' << i << ' ' << j << endl;
	File.close();
}

//备注：下面的密度和温度需要考虑之前计算时，是否已经折合过，如果折合过则下面密度不用乘Rc,温度同理
double EosPressure(const double Den)
{
	//double Pr0 = -1, Den = Density, TT = T;
	double Pr0 = -1;// Den = Density*Rc, TT = Tr*Tc;
	switch (MModel)
	{
	case MP_CPVDW: Pr0 = T * Den / (D(1) - B*Den) - A*Sq(Den);  break;

	case MP_CPPRW: Pr0 = T * Den / (D(1) - B*Den) - A*Sq(Den) / (D(1) + B*Den * 2 - Sq(B*Den));  break;

	}
	return Pr0;
}

//*************************************************************************************************
//计算界面厚度,由气相到液相提供60个格点数据,界面两侧大约各30个数据;
void GetThickness(const double Dens[], double & Width)
{
	//计算两相界面位置: (1)将格点加密后, 求等摩尔面的位置;  (2)用插值法求半密度点位置;  
	//计算过渡区的宽度: (1)用界面点的密度导数求宽度;        (2)用双曲正切函数拟合最佳宽度;
	//界面位置用等摩尔法求, 理论依据比较好;   用高次拉格朗日插值无法精确求得界面点的导数;
	double DenG = Dens[0], DenL = Dens[60];
	double MidDen = (DenG + DenL) / 2, MidPos = 0;
	for (int i = 0; i <= 60; ++i)
	{
		if (Dens[i] <= MidDen && Dens[i + 1]>MidDen)
		{
			//MidPos = D(i) + Cubic( MidDen, Dens[i-1], Dens[i], Dens[i+1], Dens[i+2] );
			MidPos = D(i) + (MidDen - Dens[i]) / (Dens[i + 1] - Dens[i]);
			break;
		}
	}

	double Diff = 10000;
	for (double W = 0.1; W <= 25; W += 0.01)
	{
		double d = 0;
		for (int i = int(MidPos - 25); i <= int(MidPos + 25 + 1); ++i)
		{
			double Den = (DenG + DenL) / 2 - (DenG - DenL) / 2 * tanh((D(i) - MidPos) * 2 / W);
			d += Sq(Den - Dens[i]);
		}

		if (d < Diff)
		{
			Diff = d;
			Width = W;
		}
	}
}


//double Sigma_IntegralMethod() 
//{
//	GridIndex;  LineIndex;
//
//	double sigma = 0;
//	for (int i = 0; i <= DX / 2; ++i) 
//	{
//		int j = DY / 2, k = DZ / 2;
//		Define_ijk5;
//		sigma += Ka * (GradY5(Dens) * GradY5(Dens) - GradX5(Dens) * GradX5(Dens));
//	}
//	return sigma;
//}
void CalcMacroCPU()
{
    Mass = 0;
    FOR_iDX_jDY_kDZ_Fluid 
    {
		double Den = Dens[I(i, j, k)];
		if( Den!=Den || Den<0 ) 
		{
			err_den = true;
			ofstream errorfile("data/error_Density.txt", ios::app);
			errorfile << "Density: "<< NowStep << " " << i << " " << j << " " << k << " Den = " << Den << endl;
			errorfile.close();
		}
		Mass += Den; //求总质量
	}

	if(err_den) 
	{
		cout << " Error Density! " << endl;
		NowStep = AllStep;
	}

	// 求最大速度
	MaxSpeed = 0;
	FOR_iDX_jDY_kDZ_Fluid {
		double Mod = Sq(MVx[I(i, j, k)]) + Sq(MVy[I(i, j, k)]) + Sq(MVz[I(i, j, k)]);
		if (Mod > MaxSpeed) {
			MaxSpeed = Mod;
		}
	}
	MaxSpeed = sqrt(MaxSpeed);
}

void ShowData()
{
	//打印表头	
    if (NowStep == 0) 
	{
		cout << "计算Pt-Chem曲线的亲水段数据" <<endl;
        cout << "程序参数设置：" << endl;
        cout << "Multiphase: " << Name(MModel) << "    DX=" << DX << "    DY=" << DY << "    DZ=" << DZ << "    Tau=" << Tau << "    Tr=" << Tr << "    DenG=" << DenG << "    DenL=" << DenL << endl;
        cout << "Radius=" << Radius 
            << "    Width=" << Width 
            << "    Ka=" << Ka 
            << "    K=" << K 
            << "    BasePt=" << BasePt 
			<< "  	AllStep=" << AllStep
            << endl;

		//化学势边界计算方式
#ifdef COMPLEXCHEMBOUNDARY
		cout << "化学势边界条件：逐层计算" << endl;
#else
		cout << "化学势边界条件：多层设置为相同" << endl;
#endif
		//化学势计算方式：
		cout << "ChemicalPotential Calculation Method: ";
#ifdef PTCALCFIVEPOINT   //五点
			cout << "FivePoint" << endl;
#elif PTCALCSEVENPOINT //七点
			cout << "SevenPoint" << endl;
#else
			cout << "Error:  Please define the Pote calculation method!" << endl;
#endif

#ifdef CONTACTANGLE	/*------------------------------------计算接触角相关------------------------------------*/
        //液滴的位置
        // cout << "LiquidDrop Position: " 
        //     << "    X=" << DropletPosX 
        //     << "    Y=" << DropletPosY 
        //     << "    Z=" << DropletPosZ 
        //     << endl;


        cout << endl;

        cout << "NowStep" 
        << "    Mass" 
        << "    Den(DX/2 , 0 , 3)" 
        << "    Den(DX/2, DY/2/8, 3)" 
        << "    Den(DX/2, DY/2/4, 3)" 
        << "    MaxSpeed"
        //<< "    Temperature(DX/2, DY/2, DZ*3/4)"
        << "    CaLeft"
        << "    CaRight"
        << "    CaCap"
        << "    XLeft"
        << "    XRight"
        << "    StepTime"  
        << endl;
        cout << "----------------------------------------------------------------------------------------------------" << endl;
    }
    
    cout << setw(9) << NowStep 
    << "    " << setiosflags(ios::fixed) << setprecision(12) << Mass 
    << "    " << setprecision(16) << Dens[I(DX / 2, 0, 3)] /Rc
	<< "    " << Dens[I(DX / 2, DY / 2 / 8, 3)] /Rc
	<< "    " << Dens[I(DX / 2, DY / 2 / 4, 3)] /Rc
    << "    " << setprecision(6)  << MaxSpeed 
    //<< "    " << Grid[DX / 2][DY / 2][DZ * 3/ 4].Te
    << "    " << CaLeft
    << "    " << CaRight
    << "    " << CaCap
    << "    " << XLeft
    << "    " << XRight
    << "    " << TimeInterval()
    << endl;
#endif
}

void SaveContactAngle()
{
	ofstream File;
	char FileName[256];
    sprintf(FileName, "data/CP%s_MRT_Tau=%2.1f_FIELD%dx%dx%d_ContactAngle_R%3.1f_Tr%3.1f.txt", Name(MModel) + 3, Tau, DX, DY, DZ, Radius, Tr);
	//if (0 == ContactAngleFileHeaderFlag && NowStep == 1)
	if(No == 0 && NowStep == 0)
	{		
        //ContactAngleFileHeaderFlag = 1;

		File.open(FileName, ios::app); //File << "Droprad   BasePt   NowStep   NowStep*DimTime   CaLeft   CaRight  CaCap  XLeft   XRight   L   Mx   My   Mz   My2   Mz2" << endl; //#
		File << "Droprad" << "  " 
            << "BasePt" << "  "
            << "NowStep" << "  "
            //<< "NowStep*DimTime" << "  "
            << "CaLeft" << "  "
            << "CaRight" << "  "
            << "CaCap" << "  "
            << "XLeft" << "  "
            << "XRight" << "  "
            << endl;
		File.close();
	}

	File.open(FileName, ios::app); //File << Radius << "   " << BasePt << "   " << NowStep << "   " << NowStep*DimTime << "   " << CaLeft << "   " << CaRight << "    " << CaCap << "   " << XLeft << "   " << XRight << "   " << (XRight - XLeft) * DimLength << "   " << Mx << "   " << My * DimLength << "   " << Mz * DimLength << "   " << My2 * DimLength << "   " << Mz2 * DimLength << endl; File.close();  //#
    File << Radius << "  " 
        << BasePt << "  " 
        << NowStep << "  " 
     //   << NowStep * DimTime << "  "
        << CaLeft << "  "
        << CaRight << "  "
        << CaCap << "  "
        << XLeft << "  "
        << XRight << "  "
        << endl;
    File.close(); 
}
