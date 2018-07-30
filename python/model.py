# Import used libraries.
import torch


class vel_regressor(torch.nn.Module):
    def __init__(self,Nin=6,Nout=1,Nlinear=112*60):
        super(vel_regressor, self).__init__()
    
        self.model1 = torch.nn.Sequential(
        torch.nn.Conv1d(Nin,60,kernel_size=3,stride=1,groups=Nin),
        torch.nn.ReLU(),
        #torch.nn.MaxPool1d(10, stride=5),
        torch.nn.Conv1d(60,120,kernel_size=3,stride=1,groups=Nin),
        torch.nn.ReLU(),
        #torch.nn.MaxPool1d(10, stride=2),
        torch.nn.Conv1d(120,240,kernel_size=3,stride=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool1d(10, stride=6),
        )
        self.model2=model2=torch.nn.Sequential(
        torch.nn.Linear(Nlinear, 10*40),
        torch.nn.ReLU(),
        torch.nn.Linear(10*40, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, Nout)
        )

        
        

    def forward(self, x):
        x = self.model1(x)
        x = x.view(x.size(0), -1)
        #print(x)
        x = self.model2(x)
        return x