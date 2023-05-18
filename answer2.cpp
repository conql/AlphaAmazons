#pragma GCC target("popcnt")
#pragma GCC optimize("unroll-loops")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2")
#include <random>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <cassert>
// #define DEBUG

typedef unsigned long long u64;
typedef unsigned u32;
typedef unsigned char u8;

double T = 1;
inline void checkTime()
{
	static u32 cnt = 0;
	if (++cnt % (1u << 14) == 0 && double(clock()) > CLOCKS_PER_SEC * T)
	{
		exit(0);
	}
}
using std::cin;
using std::cout;
inline int print(u64 x)
{
	for (int i = 0; i < 64; ++i)
	{
		cout << (x >> i & 1);
		if (i % 8 == 7)
			cout.put(10);
	}
	cout << std::endl;
	return 0;
}

namespace dir
{
	static constexpr u64 W = 9187201950435737471ull, E = 18374403900871474942ull, S = 18446744073709551360ull, N = 72057594037927935ull, SE = S & E,
						 SW = S & W,
						 NE = N & E,
						 NW = N & W;

	static constexpr u64 W2 = W >> 1 & W, W4 = W2 >> 2 & W2;
	static constexpr u64 E2 = E << 1 & E, E4 = E2 << 2 & E2;
	static constexpr u64 S2 = S >> 8 & S, S4 = S2 >> 16 & S2;
	static constexpr u64 N2 = N << 8 & N, N4 = N2 << 16 & N2;
	static constexpr u64 SE2 = SE >> 9 & SE, SE4 = SE2 >> 18 & SE2;
	static constexpr u64 SW2 = SW << 9 & SW, SW4 = SW2 << 18 & SW2;
	static constexpr u64 NE2 = NE >> 7 & NE, NE4 = NE2 >> 14 & NE2;
	static constexpr u64 NW2 = NW << 7 & NW, NW4 = NW2 << 14 & NW2;
}

static std::mt19937 gen(114514);

const int N = 8;

inline void up(double &x, double y)
{
	if (x < y)
		x = y;
}
inline bool good(int x, int y) { return (u32)x < N && (u32)y < N; }
inline int popc(u64 x) { return __builtin_popcountll(x); }

inline void flip(u64 &s, int x, int y) { s ^= (u64)1 << (x * 8 + y); }
inline int id(int x, int y) { return x * N | y; }
inline int row(u32 x) { return x / 8; }
inline int col(u32 x) { return x % 8; }

u64 edge[64], path[64][64];
u64 P0[64], P1[64], P2[64], P3[64];

inline void init()
{
	for (int i = 0; i != 64; ++i)
	{
		const int x0 = row(i), y0 = col(i);
		for (auto x : {-1, 0, 1})
			for (auto y : {-1, 0, 1})
				if (x || y)
				{
					u64 e = 0;
					for (int a = x0, b = y0;;)
					{
						a += x, b += y;
						if ((u32)a >= 8 || (u32)b >= 8)
							break;
						edge[i] |= 1ull << id(a, b);
						path[i][id(a, b)] = e;
						e |= 1ull << id(a, b);
					}
				}
	}
	for (int i = 0; i < 64; ++i)
	{
		for (int j = 0; j < 64; ++j)
			if (j != i)
			{
				P0[i] |= u64(row(i) == row(j)) << j;
				P1[i] |= u64(col(i) == col(j)) << j;
				P2[i] |= u64(col(i) + row(i) == col(j) + row(j)) << j;
				P3[i] |= u64(col(i) - row(i) == col(j) - row(j)) << j;
			}
	}
}
inline void qwq(u64 &x, int p)
{
	u64 large = x >> p;
	large = (large & -large) - 1;
	u64 small = x & ((1ull << p) - 1);
	if (small == 0)
	{
		x = ((1ull << p) - 1) | large << p;
	}
	else
	{
		int bit = 63 - __builtin_clzll(small);
		x = ((1ull << p) - (2ull << bit)) | large << p;
	}
}
inline u64 canput(int id, u64 c)
{
	u64 a0 = c & P0[id];
	u64 a1 = c & P1[id];
	u64 a2 = c & P2[id];
	u64 a3 = c & P3[id];
	qwq(a0, id);
	qwq(a1, id);
	qwq(a2, id);
	qwq(a3, id);
	return (a0 & P0[id]) | (a1 & P1[id]) | (a2 & P2[id]) | (a3 & P3[id]);
}
struct move
{
	u8 start, go, arrow;
};
inline int getSpace(u64 a, u64 c)
{
	using namespace dir;
	c ^= a;
	for (u64 b;; a = b)
	{
		b = a | (a >> 1 & W) | (a << 1 & E) | (a >> 8 & dir::N) | (a << 8 & S) |
			(a >> 9 & NW) | (a << 9 & SE) | (a >> 7 & NE) | (a << 7 & SW);
		b &= ~c;
		if (b == a)
			break;
	}
	return popc(a);
}

static constexpr size_t size = 1 << 20;
static move buf[size], *ed = buf + size;
inline move *alloc(size_t L)
{
	return ed -= L;
}
struct vector_move
{
	move *b, *e;
	inline const move *begin() const { return b; }
	inline const move *end() const { return e; }
	inline vector_move(move *L, move *R)
	{
		e = ed, b = alloc(R - L);
		memcpy(b, L, sizeof(move) * (R - L));
	}
	inline ~vector_move()
	{
#ifdef DEBUG
		if (b == ed)
#endif
			ed = e;
#ifdef DEBUG
		else
			exit(1);
#endif
	}
};
struct board;
inline double eval(const board &x);
struct board
{
	u64 a, b, c;
	inline board() {}
	inline board(u64 A, u64 B, u64 C) : a(A), b(B), c(C) {}
	inline board Move(int p0, int p1, int p2) const
	{
#ifdef DEBUG
		assert((u32)p0 < 64u);
		assert((u32)p1 < 64u);
		assert((u32)p2 < 64u);
		assert((a >> p0 & 1) == 1);
		assert((c >> p1 & 1) == 0);
		assert((c >> p2 & 1) == 0 || p0 == p2);
		assert((edge[p0] >> p1 & 1) == 1);
		assert((edge[p1] >> p2 & 1) == 1);
		assert((c & path[p0][p1]) == 0);
		assert(((c ^ (1ull << p0)) & path[p1][p2]) == 0);
#endif
		return board(b, a ^ (1ull << p0) ^ (1ull << p1), c ^ (1ull << p0) ^ (1ull << p1) ^ (1ull << p2));
	}
	inline board Move(const move &x) const
	{
		return this->Move(x.start, x.go, x.arrow);
	}
	inline double eval() const
	{
		return ::eval(*this);
	}
	inline vector_move getmoves() const
	{
		static constexpr int cnt = 5000;
		static move arr[cnt];
		move *stack = arr;
		for (u64 x = a; x; x &= x - 1)
		{
			int id = __builtin_ctzll(x);
			for (u64 y = canput(id, c); y; y &= y - 1)
			{
				int go = __builtin_ctzll(y);
				const u64 cc = c ^ (1ull << id);
				for (u64 z = canput(go, cc); z; z &= z - 1)
				{
					int arrow = __builtin_ctzll(z);
					stack->start = id;
					stack->go = go;
					stack->arrow = arrow;
					++stack;
				}
			}
		}
		return vector_move(arr, stack);
	}
	inline double bestmoves(double beta) const
	{
		double ans = -1e9;
		for (u64 x = a; x; x &= x - 1)
		{
			int id = __builtin_ctzll(x);
			for (u64 y = canput(id, c); y; y &= y - 1)
			{
				int go = __builtin_ctzll(y);
				const u64 cc = c ^ (1ull << id);
				for (u64 z = canput(go, cc); z; z &= z - 1)
				{
					int arrow = __builtin_ctzll(z);
					static move P;
					P.start = id;
					P.go = go;
					P.arrow = arrow;
					up(ans, 1 - Move(P).eval());
					if (ans > beta)
						return ans;
				}
			}
		}
		return ans;
	}
	inline board reverse() const
	{
		return board(b, a, c);
	}
} EP;

struct DisArr
{
	u64 a[38];
	int len;
	inline void clear()
	{
		memset(a, 0, (len + 1) << 3);
	}
};
inline void KingMoveBfs(const board &x, DisArr &A)
{
	using namespace dir;
	*A.a = x.a;
	const u64 P = x.c ^ x.a;
	int &i = A.len = 0;
	do
	{
		++i;
		u64 a = A.a[i - 1];
		a = a | (a >> 1 & W) | (a << 1 & E) | (a >> 8 & dir::N) | (a << 8 & S) |
			(a >> 9 & NW) | (a << 9 & SE) | (a >> 7 & NE) | (a << 7 & SW);
		A.a[i] = a & ~P;

	} while (A.a[i] != A.a[i - 1]);
	for (int x = i - 1; x >= 1; --x)
		A.a[x] ^= A.a[x - 1];
}
inline void KingMoveBfs(const board &x, DisArr &A, DisArr &B)
{
	KingMoveBfs(x, A), KingMoveBfs(x.reverse(), B);
}
inline void QueenMoveBfs(const board &x, DisArr &A)
{
	using namespace dir;
	*A.a = x.a;
	const u64 P = x.c ^ x.a;
	int &i = A.len = 0;
	do
	{
		++i;
		u64 a = A.a[i - 1], result = a;
#define OP(shift, num, forand)                   \
	{                                            \
		u64 z = a;                               \
		result |= z = z shift num & forand & ~P; \
		result |= z = z shift num & forand & ~P; \
		result |= z = z shift num & forand & ~P; \
		result |= z = z shift num & forand & ~P; \
		result |= z = z shift num & forand & ~P; \
		result |= z = z shift num & forand & ~P; \
		result |= z = z shift num & forand & ~P; \
	}
		OP(>>, 1, W);
		OP(<<, 1, E);
		OP(>>, 8, dir::N);
		OP(<<, 8, S);
		OP(>>, 9, NW);
		OP(<<, 9, SE);
		OP(>>, 7, NE);
		OP(<<, 7, SW);
#undef OP
		A.a[i] = result & ~P;

	} while (A.a[i] != A.a[i - 1]);
	for (int x = i - 1; x >= 1; --x)
		A.a[x] ^= A.a[x - 1];
}
inline void QueenMoveBfs(const board &x, DisArr &A, DisArr &B)
{
	QueenMoveBfs(x, A), QueenMoveBfs(x.reverse(), B);
}
inline double mobility(const board &x)
{
	using namespace dir;
	u64 a0 = ~x.c >> 1 & W;
	u64 a1 = ~x.c << 1 & E;
	u64 a2 = ~x.c >> 8 & dir::N;
	u64 a3 = ~x.c << 8 & S;
	u64 a4 = ~x.c >> 9 & NW;
	u64 a5 = ~x.c << 9 & SE;
	u64 a6 = ~x.c >> 7 & NE;
	u64 a7 = ~x.c << 7 & SW;
	double s0 = 0, s1 = 0;
#define OP(shift, num, forand, s, P, ad)                                                                                                                                              \
	{                                                                                                                                                                                 \
		u64 z = s shift num & forand & ~P;                                                                                                                                            \
		ad += (popc(z & a0) + popc(z & a1) + popc(z & a2) + popc(z & a3) + popc(z & a4) + popc(z & a5) + popc(z & a6) + popc(z & a7)) / double(1), z = z shift num & forand & ~P;     \
		ad += (popc(z & a0) + popc(z & a1) + popc(z & a2) + popc(z & a3) + popc(z & a4) + popc(z & a5) + popc(z & a6) + popc(z & a7)) / double(2), z = z shift num & forand & ~P;     \
		for (int i = 3; z; ++i)                                                                                                                                                       \
			ad += (popc(z & a0) + popc(z & a1) + popc(z & a2) + popc(z & a3) + popc(z & a4) + popc(z & a5) + popc(z & a6) + popc(z & a7)) / double(i), z = z shift num & forand & ~P; \
	}
	u64 P = x.c;
	OP(>>, 1, W, x.a, P, s0);
	OP(<<, 1, E, x.a, P, s0);
	OP(>>, 8, dir::N, x.a, P, s0);
	OP(<<, 8, S, x.a, P, s0);
	OP(>>, 9, NW, x.a, P, s0);
	OP(<<, 9, SE, x.a, P, s0);
	OP(>>, 7, NE, x.a, P, s0);
	OP(<<, 7, SW, x.a, P, s0);

	OP(>>, 1, W, x.b, P, s1);
	OP(<<, 1, E, x.b, P, s1);
	OP(>>, 8, dir::N, x.b, P, s1);
	OP(<<, 8, S, x.b, P, s1);
	OP(>>, 9, NW, x.b, P, s1);
	OP(<<, 9, SE, x.b, P, s1);
	OP(>>, 7, NE, x.b, P, s1);
	OP(<<, 7, SW, x.b, P, s1);

#undef OP
	return s0 - s1;
}

inline int calc(const DisArr &x, const DisArr &y, u64 state)
{
	u64 map = 0;
	int res = 0;
	for (int i = 1; i < x.len && i < y.len; ++i)
	{
		res -= popc(map & x.a[i]);
		res += popc(map & y.a[i]);
		map |= x.a[i] | y.a[i];
	}
	for (int i = x.len; i < y.len; ++i)
	{
		res += popc(map & y.a[i]);
		map |= y.a[i];
	}
	for (int i = y.len; i < x.len; ++i)
	{
		res -= popc(map & x.a[i]);
		map |= x.a[i];
	}
	return res + popc(map & (state ^ y.a[y.len])) - popc(map & (state ^ x.a[x.len]));
}
namespace tables
{
	double arr[][6] = {
		{-0.035135168582201004, -0.03396248072385788, 0.08273947238922119, 0.08377216011285782, 0.037860769778490067, 0.4445617198944092},
		{-0.00795679446309805, 0.03826093673706055, -0.04300055652856827, 0.13918864727020264, 0.021546466276049614, -0.0748395025730133},
		{-0.014996325597167015, 0.030489521101117134, -0.026632605120539665, 0.10908545553684235, 0.030750950798392296, -0.030947627499699593},
		{0.0001379172899760306, -0.025661280378699303, 0.07981404662132263, 0.09567344188690186, 0.027800239622592926, 0.4343096911907196},
		{0.003942048642784357, -0.012665736488997936, 0.06189047545194626, 0.0973064973950386, 0.026430297642946243, 0.22047391533851624},
		{0.006807683035731316, -0.03217008709907532, 0.08798390626907349, 0.11188705265522003, 0.03374253958463669, 0.36134710907936096},
		{0.012731116265058517, -0.03741123154759407, 0.09410484880208969, 0.1335466504096985, 0.024877874180674553, 0.40032923221588135},
		{0.012298612855374813, -0.012371799908578396, 0.052135076373815536, 0.13047397136688232, 0.028690574690699577, 0.16860519349575043},
		{0.010649841278791428, -0.03231256082653999, 0.10024367272853851, 0.13056616485118866, 0.03187330812215805, 0.35186249017715454},
		{0.00837423000484705, -0.010549443773925304, 0.07099789381027222, 0.12282957136631012, 0.03179285675287247, 0.1589813381433487},
		{0.005355560686439276, -0.009044983424246311, 0.07949510961771011, 0.1333647072315216, 0.020389290526509285, 0.1579679697751999},
		{0.010674690827727318, -0.0037427651695907116, 0.06529445946216583, 0.1313590258359909, 0.012989195995032787, 0.17210964858531952},
		{0.015536517836153507, 0.009275969117879868, 0.049864184111356735, 0.13854867219924927, 0.017690081149339676, 0.002616666257381439},
		{0.023774918168783188, 0.005980607122182846, 0.05142921954393387, 0.14237751066684723, 0.004873865284025669, 0.06856407225131989},
		{0.03149674832820892, 0.008815981447696686, 0.04838938266038895, 0.13119739294052124, 0.01889057457447052, -0.0178539976477623},
		{0.038709428161382675, 0.023727236315608025, 0.024715593084692955, 0.12618811428546906, 0.009036652743816376, -0.011969412676990032},
		{0.04421858489513397, 0.026028841733932495, 0.032979585230350494, 0.118276447057724, 0.014974003657698631, -0.12384149432182312},
		{0.042966656386852264, 0.03693069517612457, 0.027083303779363632, 0.11631295830011368, 0.019244173541665077, -0.027017179876565933},
		{0.054176799952983856, 0.035372450947761536, 0.03293950483202934, 0.10154443979263306, 0.013590249232947826, -0.18943949043750763},
		{0.05432905629277229, 0.0465015210211277, 0.024094300344586372, 0.09414323419332504, 0.021908367052674294, -0.06966306269168854},
		{0.06536839157342911, 0.029863806441426277, 0.051898617297410965, 0.0842253714799881, 0.023366985842585564, -0.14950041472911835},
		{0.060640640556812286, 0.047984782606363297, 0.02703857235610485, 0.08638651669025421, 0.031321510672569275, -0.07337093353271484},
		{0.058226194232702255, 0.046266041696071625, 0.018133902922272682, 0.09641703218221664, 0.03714189678430557, -0.14568787813186646},
		{0.06938368827104568, 0.05292125791311264, 0.027041025459766388, 0.0883697047829628, 0.043525658547878265, -0.10619260370731354},
		{0.0851794108748436, 0.056678012013435364, 0.018657652661204338, 0.0851815938949585, 0.038640279322862625, -0.18063239753246307},
		{0.08329196274280548, 0.0759916678071022, 0.015736404806375504, 0.07772944122552872, 0.07035521417856216, -0.09182342141866684},
		{0.07090955227613449, 0.09454822540283203, -0.015821417793631554, 0.09409584105014801, 0.13349442183971405, -0.22048918902873993},
		{0.06693396717309952, 0.10030302405357361, -0.004830216057598591, 0.10458771884441376, 0.1755196452140808, -0.07642731815576553},
		{0.0851433053612709, 0.10500379651784897, -0.00039774845936335623, 0.10763891041278839, 0.21350035071372986, -0.17134110629558563},
		{0.09496818482875824, 0.14105741679668427, -0.02626052126288414, 0.11477160453796387, 0.2728651165962219, -0.08919291943311691},
		{0.08454374223947525, 0.18916209042072296, -0.042582038789987564, 0.12307386100292206, 0.3566949963569641, -0.2549653649330139},
		{0.03392213582992554, 0.24692672491073608, -0.06186968833208084, 0.15846525132656097, 0.4889538884162903, -0.13768702745437622},
		{0.09719052165746689, 0.2646160423755646, -0.09884422272443771, 0.17642264068126678, 0.5493039488792419, -0.24980612099170685},
		{0.14186908304691315, 0.26009488105773926, -0.0934523344039917, 0.18925561010837555, 0.5786257982254028, -0.27276527881622314},
		{0.13207440078258514, 0.34693968296051025, -0.14419978857040405, 0.20928022265434265, 0.6583712697029114, -0.30505016446113586},
		{0.2441231906414032, 0.32330313324928284, -0.08880028128623962, 0.20839904248714447, 0.5671485662460327, -0.404548317193985},
		{-0.08759017288684845, 0.6787862777709961, -0.11531388014554977, 0.2509157359600067, 0.6201470494270325, -0.3662388324737549},
		{-0.17256882786750793, 0.8728899359703064, -0.1190086081624031, 0.32024940848350525, 0.6679149270057678, -0.504874050617218},
		{0.10061004757881165, 1.056464433670044, -0.14753708243370056, 0.12780454754829407, 0.7711073756217957, -0.5921382308006287},
		{-0.022837769240140915, 0.9750547409057617, -0.16612277925014496, 0.4281192719936371, 0.7386918663978577, -0.7177150249481201},
		{0.7156360745429993, 0.5276584029197693, -0.13439998030662537, 0.306896835565567, 0.8386046290397644, -0.7461664080619812},
		{0.677611768245697, -0.011891359463334084, -0.15832264721393585, 1.1653300523757935, 0.85899418592453, -0.8795626163482666},
		{0.6521938443183899, 0.053694795817136765, -0.19955866038799286, 1.5037997961044312, 1.0734232664108276, -1.0472180843353271},
		{0.5191304683685303, 0.580433189868927, -0.16413605213165283, 1.5544768571853638, 0.992075502872467, -1.2729535102844238},
		{0.9855523705482483, 1.1670809984207153, -0.32054752111434937, 1.3204418420791626, 0.9672726988792419, -1.6374696493148804},
		{0.9297383427619934, 1.290230631828308, -0.11891954392194748, 1.8169777393341064, 0.9424067139625549, -2.005551338195801},
		{1.5491220951080322, 1.5979832410812378, 0.28321143984794617, 1.8582204580307007, 0.060512419790029526, -2.595350503921509},
		{1.8633183240890503, 1.947853684425354, -0.3602556586265564, 1.7069785594940186, 0.4738243818283081, -2.8554534912109375},
		{1.8316141366958618, 1.7095311880111694, 0.30481910705566406, 1.9863288402557373, 0.43060997128486633, -2.7020580768585205},
		{1.441033124923706, 1.8446333408355713, 0.517808735370636, 1.9932639598846436, 0.8255653977394104, -3.290686845779419},
		{1.7869746685028076, 1.6570954322814941, 0.6800280809402466, 1.9698376655578613, 1.0062763690948486, -2.9381096363067627},
		{1.9097577333450317, 1.8296443223953247, -0.21192480623722076, 1.8030561208724976, 1.379346251487732, -3.3817524909973145},
		{1.8501962423324585, 2.1133084297180176, 1.0673166513442993, 2.073594331741333, 0.5281442403793335, -3.7167840003967285},
		{-0.07112371176481247, 0.07493814080953598, 0.2855735123157501, 0.06107020378112793, -0.1077546626329422, -4.567995548248291}};

}
inline double eval(const board &x)
{
	checkTime();

	static constexpr double fm_adv = 0.3;
	static DisArr k0, k1, q0, q1;
	KingMoveBfs(x, k0, k1);
	QueenMoveBfs(x, q0, q1);
	const u64 pos = ~x.c;
	double p0 = calc(k0, k1, pos) + popc(k0.a[1] & k1.a[1]) * fm_adv;
	double p1 = calc(q0, q1, pos) + popc(q0.a[1] & q1.a[1]) * fm_adv;
	double p2 = 0;
	double p3 = 0;
	double mob = mobility(x) * 0.1;
	const u64 reach0 = k0.a[k0.len] ^ *k0.a, reach1 = k1.a[k1.len] ^ *k1.a, range = reach0 & reach1;
	p3 += popc(reach0 & ~reach1) - popc(reach1 & ~reach0);
	for (int i = 1; i < k0.len; ++i)
		p3 -= popc(range & k0.a[i]) / 6. * i;
	for (int i = 1; i < k1.len; ++i)
		p3 += popc(range & k1.a[i]) / 6. * i;
	static double li[] = {
		2. / (1 << 0),
		2. / (1 << 1),
		2. / (1 << 2),
		2. / (1 << 3),
		2. / (1 << 4),
		2. / (1 << 5),
	};
	for (int i = 1; i < 6 && i < q0.len; ++i)
		p2 += popc(q0.a[i]) * li[i];
	for (int i = 1; i < 6 && i < q1.len; ++i)
		p2 -= popc(q1.a[i]) * li[i];
	// fprintf(stderr, "%.4lf %.4lf %.4lf %.4lf %.4lf\n", p1, p0, p2, p3, mob);
	k0.clear(), k1.clear(), q0.clear(), q1.clear();
	const int id = popc(x.c) - 9;
	const double *o = tables::arr[id];
	return 1 / (1 + std::exp(-(o[0] * p0 + o[1] * p1 + o[2] * p2 + o[3] * p3 + o[4] * mob + o[5])));
}
inline board initboard()
{
	int turnid, x0, y0, x1, y1, x2, y2;
	using std::cin;
	using std::cout;
	cin >> turnid;
	if (turnid == 1)
	{
		T *= 2;
	}
	board b(
		1ull << id(0, 2) | 1ull << id(2, 0) |
			1ull << id(5, 0) | 1ull << id(7, 2),
		1ull << id(0, 5) | 1ull << id(2, 7) |
			1ull << id(5, 7) | 1ull << id(7, 5),
		0);
	b.c = b.a | b.b;

	for (int i = 0; i < turnid; ++i)
	{
		cin >> x0 >> y0 >> x1 >> y1 >> x2 >> y2;
		if (x0 != -1)
		{
			b = b.Move(id(x0, y0), id(x1, y1), id(x2, y2));
		}
		if (i < turnid - 1)
		{
			cin >> x0 >> y0 >> x1 >> y1 >> x2 >> y2;
			b = b.Move(id(x0, y0), id(x1, y1), id(x2, y2));
		}
	}
	return b;
}
inline board initboard_new()
{
	u64 self = 0, opp = 0, block = 0;
	for (int y = 0; y < 8; y++)
	{
		for (int x = 0; x < 8; x++)
		{
			int t;
			cin >> t;

			if (t < 1 || t > 3)
				continue;

			if (t == 1)
				self |= 1ull << id(x, y);
			if (t == 2)
				opp |= 1ull << id(x, y);

			block |= 1ull << id(x, y);
		}
	}

	return board(self, opp, block);
}

int rate[] = {10, 6, 9, 5, 7, 5, 6, 4, 4, 4, 4, 4, 4, 4, 4};
int SD;
struct pr
{
	double score;
	board B;
};
inline bool cmp(const pr &x, const pr &y)
{
	return x.score < y.score;
}
int lose;
template <const int TYPE>
inline double search(const board &b, int dep, double alpha, double beta)
{
	// ans < alpha : any < alpha
	// ans in [alpha, beta] : ans
	// ans > beta : any > beta
	if (dep == SD)
		return b.eval();
	const double P = b.eval();
	const double AF = 0.7 + (dep == SD - 1) * 0.1;
	if (P + (1 - P) * (1 - AF) < alpha)
		return P;
	if (P * AF > beta)
		return P;
	if (dep == SD - 1)
	{
		lose = 1;
		return b.bestmoves(beta);
	}
	auto p = b.getmoves();
	if (p.begin() == p.end())
		return -1e8;
	static pr mem[20][10000];
	pr *A = mem[dep], *ed = A;
	for (const auto &c : p)
	{
		ed->B = b.Move(c);
		ed->score = ed->B.eval();
		++ed;
	}
	const int size = TYPE ? std::min<int>(rate[dep], ed - A) : ed - A;
	if (size < ed - A)
	{
		std::nth_element(A, A + size, ed, cmp);
		std::sort(A, A + size, cmp);
	}
	double res = alpha;
	for (int i = 0; i < size; ++i)
	{
		up(res, 1 - search<TYPE>(A[i].B, dep + 1, 1 - beta, 1 - res));
		if (res > beta)
			break;
	}
	return res;
}

struct answer_writer
{
	int ok;
	move a;
	answer_writer()
	{
	}
	~answer_writer()
	{
		if (ok)
		{
			cout << row(a.start) << ' ' << col(a.start) << ' ' << row(a.go) << ' ' << col(a.go) << ' ' << row(a.arrow) << ' ' << col(a.arrow) << '\n';
		}
		else
		{
			puts("-1 -1 -1 -1 -1 -1");
		}
	}
} answer;
template <const int TYPE>
inline void solve(int D, board b)
{
	auto p = b.getmoves();
	if (std::begin(p) == std::end(p))
	{
		return;
	}
	move a;
	double best = 1e9;
	SD = D - 1;
	lose = 0;
	std::vector<move> v(p.begin(), p.end());
	stable_sort(v.begin(), v.end(), [&](move x, move y)
				{ return b.Move(x).eval() < b.Move(y).eval(); });
	if (popc(b.c) == 8)
	{
		v.erase(remove_if(v.begin(), v.end(), [](move x)
						  { return row(x.start) >= 3; }),
				v.end());
	}
	if (popc(b.c) > 8)
		if (v.size() > 12)
			v.resize(12);

	int P = 0;
	for (auto c : v)
	{
		const double val = search<TYPE>(b.Move(c), 0, -1e9, best);
		if (val < best)
			best = val, a = c;
		//  std::cerr << "chose " << P << std::endl;
		++P;
	}
	answer.ok = 1;
	answer.a = a;
	// std::cerr << 1-best << ' ' << b.eval() << '\n';
}
int main()
{
	// freopen("./2.in", "r", stdin);

	init();
	board b = initboard_new();
	for (int i = 2; i < 15; i += 1)
	{
		solve<1>(i, b);
		// std::cerr << "search dep " << i << ' ' << double(clock()) / CLOCKS_PER_SEC << std::endl;

		if (!lose)
		{
			solve<0>(i, b);
			return 0;
		}
	}
}