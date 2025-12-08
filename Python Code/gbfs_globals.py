GAPGEN = 5

DELT = None
OMEGA = None

M = 2

visual_dir = None

data = None
label = None

trIdx = None

featNum = None
Zout = None          # redundancy graph (thresholded abs-corr)

vWeight = None       # dynamic node weights (start from Fisher)
vWeight1 = None

kNeigh = 5           # giữ để tương thích chỗ khác, nhưng không dùng cho chọn k-NN edges nữa
kNeiMatrix = None
kNeiZout = None      # = Zout trong hướng mới (để tương thích)

trData = None
trLabel = None
teData = None
teLabel = None

assiNumInside = []
Weight = None        # alias cho redundancy graph, nếu nơi khác dùng

TTT = []
TTTT = []

# logging buffers
current_run = None
run_logs = {}

# ====== NEW for node-based + complementary MI ======
miFeat = None        # MI(feature; Y), shape (featNum,)
compMat = None       # complementary matrix c_ij >= 0, shape (featNum, featNum)
alphaComp = 0.5      # trade-off khi tính node weight ở đồ thị complementary
