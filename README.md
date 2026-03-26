# Step-0: Open the anaconda prompt
conda init powershell



Start the VS Code

# Step-0.1: Exceute in Vs code
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Step-1: Setup the enviroment with required libraries
conda create -n cv_facefilter python=3.10 -y

# Step-2: Activate environment
conda activate cv_facefilter

# Step-3: Checking env is activated or not
python --version

# Step-4: Restart the VS code
CTRL + SHIFT + P  -> interpreter  -> select the cv_facefilter

# Step-5: installing required libraries
pip install mediapipe==0.10.8 opencv-python numpy