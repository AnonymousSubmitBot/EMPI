# Set Path
export $EMPI=~/EMPI_Project

#Initial Folder
rm -rf ~/anaconda3/pkgs/ncurses-6.4-h6a678d5_0
rm -rf ~/anaconda3/pkgs/_openmp_mutex-5.1-1_gnu
conda create -n EMPI -q -y python=3.10

# Install Packages from apt
sudo apt install cmake make gcc g++ libeigen3-dev libssl-dev swig git libasio-dev

# Install Python Package
cd $EMPI
conda activate EMPI
pip install -r requirements.txt

# Download and Install CMAKE
cd $EMPI
wget https://cmake.org/files/v3.22/cmake-3.22.4.tar.gz
tar xf cmake-3.22.4.tar.gz
cd cmake-3.22.4
./bootstrap --parallel=48
make -j 128
sudo make install

# Download and Install PyBind11
cd $EMPI
git clone https://github.com/pybind/pybind11.git
cd  pybind11
mkdir build
cd build
cmake ..
make check -j 128
sudo make install

# Download and Install Boost
cd $EMPI
wget https://archives.boost.io/release/1.84.0/source/boost_1_84_0.tar.gz
tar xf boost_1_84_0.tar.gz
cd boost_1_84_0
./bootstrap.sh
sudo ./b2 install --prefix=/usr toolset=gcc threading=multi

# Clear Current Boost TMP
sudo rm -r /usr/lib/x86_64-linux-gnu/cmake/Boost-1.74.0

# Compile CIMP Lib
sudo ldconfig
cd $EMPI/src/cpp/com_imp
rm -rf CMakeCache.txt CMakeFiles cmake_install.cmake Makefile CTestTestfile.cmake _deps COMIMP
cmake -DCMAKE_BUILD_TYPE=Release && make -j 128

# Clear Source Files
sudo rm -rf $EMPI/boost_1_84_0* $EMPI/cmake-3.22.4* $EMPI/pybind11

# Initial CK
ck pull repo:ck-env
ck pull repo:ck-autotuning
ck pull repo:ctuning-programs
ck pull repo:ctuning-datasets-min

# Create Mem TMP for CK
sudo mkdir /tmp_ck
sudo chown -R ${USER} /tmp_ck
sudo mount -t tmpfs -o size=32G tmpfs /tmp_ck
rm -rf /tmp_ck/*
cp -r $EMPI/data/dataset/compiler_args/CK ~/
cp -r $EMPI/data/dataset/compiler_args/CK-TOOLS ~/
cp -r $EMPI/data/dataset/compiler_args/* /tmp_ck/


wget https://mirrors.aliyun.com/golang/go1.24.1.linux-amd64.tar.gz
mkdir GOLANG
tar -C ./GOLANG/ -xzf go1.24.1.linux-amd64.tar.gz
cd GOLANG/go

# If you use bash, you should modify ~/.zshrc to ~/.bashrc,
# If you use fish, you should modify ~/.zshrc to ~/.fishrc,
echo "export GOPATH=$EMPI/GOLANG/go" >> ~/.zshrc
echo "export PATH=$PATH:$GOPATH/bin/" >> ~/.zshrc
source ~/.zshrc