import os
import pickle
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_data
from crank_nicolson import make_data
from haven import haven_utils as hu
from lstm_model import LSTMModel
from gilr_model import GILRModel
from attn_model import AttentionModel
from plotting import plot_experiment
from run import Run


@dataclass
class Experiment:
    n_x: int
    n_t: int
    c_x: int
    c_t: int
    r: float
    u_min: float
    u_max: float
    num_samples: int
    num_epochs: int
    model: any
    batch_size: any
    runs: list[Run] = field(default_factory=list)
    data: any = None

    def run_experiment(self, plot_only, path, device):
        experiment_id = hu.hash_dict({"experiment": self})
        if plot_only is False:
            ### Create data
            np.random.seed(0)
            u = make_data(
                self.n_x,
                self.n_t,
                self.c_x,
                self.c_t,
                self.r,
                self.u_min,
                self.u_max,
                self.num_samples,
            )
            self.data = torch.tensor(u).to(torch.float32)
            ### Iterate runs
            for run_num, run in enumerate(self.runs):
                torch.manual_seed(0)
                print("-----Run " + str(run_num + 1) + "-----")
                print("Number of data points: " + str(run.num_samples * (self.n_t - 1)))
                run.training_data = self.data[:, 0:run.num_samples, :]
                ## Pick model
                if self.model == "LSTM":
                    run.model = LSTMModel(
                        input_size=self.n_x,
                        lstm_size=run.lstm_size,
                        fnn_size=run.fnn_size,
                        output_size=self.n_x,
                    ).to(device)
                if self.model == "GILR":
                    run.model = GILRModel(
                        input_size=self.n_x,
                        lstm_size=run.lstm_size,
                        fnn_size=run.fnn_size,
                        output_size=self.n_x,
                    ).to(device)
                if self.model == "Attention":
                    run.model = AttentionModel(
                        input_size=self.n_x,
                        attn_size=run.lstm_size,
                        fnn_size=run.fnn_size,
                        output_size=self.n_x,
                        max_length=self.n_t,
                        device=device,
                    ).to(device)
                ### Train model
                run.num_params = sum(p.numel() for p in run.model.parameters())
                print("Number of parameters: " + str(run.num_params))
                run.training_losses = self.train(run, device)
            with open(
                os.path.join(path, "experiments", str(experiment_id) + ".pkl"),
                "wb",
            ) as f:
                pickle.dump({"experiment": self}, f)
            f.close()
        with open(
            os.path.join(path, "experiments", str(experiment_id) + ".pkl"), "rb"
        ) as f:
            experiment = pickle.load(f)
        f.close()
        ### Plot results
        print("Plot Experiment")
        plot_experiment(experiment["experiment"], path)

    def train(self, run, device):
        if self.batch_size == "full":
            batch_size = int(run.num_samples)
        else:
            batch_size = int(self.batch_size)

        criterion = nn.MSELoss()
        step_size = 0.001
        optimizer = optim.Adam(run.model.parameters(), lr=step_size)
        trainloader = torch_data.DataLoader(
            run.training_data, batch_size=batch_size, shuffle=False
        )
        training_losses = []
        for epoch in range(self.num_epochs):
            for _, batch in enumerate(trainloader):
                batch = batch.to(device)
                optimizer.zero_grad()
                output = run.model(batch[:-1, :, :])
                loss = criterion(output, batch[1:, :, :])
                loss.backward()
                optimizer.step()

            if epoch == 0:
                training_loss = self.compute_full_loss(
                    run, device, batch_size
                )
                print(
                    f"Initial Training Loss: {training_loss}"
                )
                training_losses.append(training_loss)
            if epoch == self.num_epochs - 1:
                training_loss = self.compute_full_loss(
                    run, device, batch_size
                )
                print(
                    f"Final Training Loss: {training_loss}"
                )
                training_losses.append(training_loss)

        return training_losses

    def compute_full_loss(self, run, device, batch_size):
        criterion = nn.MSELoss()

        training_loss = 0
        for batch in torch_data.DataLoader(
            run.training_data, batch_size=batch_size, shuffle=False
        ):
            batch = batch.to(device)
            output = run.model(batch[:-1, :, :])
            loss = criterion(output, batch[1:, :, :])
            training_loss += loss
        training_loss = training_loss.item()

        return training_loss
