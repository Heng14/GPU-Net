import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net,MixU_Net,MixAttU_Net, MixR2U_Net, MixR2AttU_Net, GhostU_Net, GhostU_Net1, GhostU_Net2
import csv
from skimage.io import imsave, imread
import matplotlib.pyplot as plt
from thop import profile
from torchstat import stat

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


class Solver(object):
	def __init__(self, config, train_loader, valid_loader, test_loader):

		# Data loader
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.test_loader = test_loader

		# Models
		self.unet = None
		self.optimizer = None
		self.img_ch = config.img_ch
		self.output_ch = config.output_ch
		self.criterion = torch.nn.BCELoss()
		self.augmentation_prob = config.augmentation_prob

		# Hyper-parameters
		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2

		# Training settings
		self.num_epochs = config.num_epochs
		self.num_epochs_decay = config.num_epochs_decay
		self.epochs_decay_rate = config.epochs_decay_rate
		self.batch_size = config.batch_size


		# Step size
		self.log_step = config.log_step
		self.val_step = config.val_step

		# Path
		self.model_path = config.model_path
		self.result_path = config.result_path
		self.mode = config.mode

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model_type = config.model_type
		self.t = config.t
		self.build_model()

	def build_model(self):
		"""Build generator and discriminator."""
		if self.model_type =='U_Net':
			self.unet = U_Net(img_ch=self.img_ch,output_ch=1)
		elif self.model_type =='R2U_Net':
			self.unet = R2U_Net(img_ch=self.img_ch,output_ch=1,t=self.t)
		elif self.model_type =='AttU_Net':
			self.unet = AttU_Net(img_ch=self.img_ch,output_ch=1)
		elif self.model_type == 'R2AttU_Net':
			self.unet = R2AttU_Net(img_ch=self.img_ch,output_ch=1,t=self.t)
		elif self.model_type =='MixU_Net':
			self.unet = MixU_Net(img_ch=self.img_ch,output_ch=1)
		elif self.model_type =='MixAttU_Net':
			self.unet = MixAttU_Net(img_ch=self.img_ch,output_ch=1)
		elif self.model_type =='MixR2U_Net':
			self.unet = MixR2U_Net(img_ch=self.img_ch,output_ch=1)
		elif self.model_type =='MixR2AttU_Net':
			self.unet = MixR2AttU_Net(img_ch=self.img_ch,output_ch=1)
		elif self.model_type =='GhostU_Net':
			self.unet = GhostU_Net(img_ch=self.img_ch,output_ch=1)
		elif self.model_type =='GhostU_Net1':
			self.unet = GhostU_Net1(img_ch=self.img_ch,output_ch=1)
		elif self.model_type =='GhostU_Net2':
			self.unet = GhostU_Net2(img_ch=self.img_ch,output_ch=1)

		#pytorch_total_params = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
		#print (pytorch_total_params)
		#raise
		self.optimizer = optim.Adam(list(self.unet.parameters()),
									  self.lr, [self.beta1, self.beta2])
		self.unet.to(self.device)

		# self.print_network(self.unet, self.model_type)

	def print_network(self, model, name):
		"""Print out the network information."""
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(model)
		print(name)
		print("The number of parameters: {}".format(num_params))

	def to_data(self, x):
		"""Convert variable to tensor."""
		if torch.cuda.is_available():
			x = x.cpu()
		return x.data

	def update_lr(self, g_lr, d_lr):
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = lr

	def reset_grad(self):
		"""Zero the gradient buffers."""
		self.unet.zero_grad()

	def compute_accuracy(self,SR,GT):
		SR_flat = SR.view(-1)
		GT_flat = GT.view(-1)

		acc = GT_flat.data.cpu()==(SR_flat.data.cpu()>0.5)

	def tensor2img(self,x):
		img = (x[:,0,:,:]>x[:,1,:,:]).float()
		img = img*255
		return img


	def train(self):
		"""Train encoder, generator and discriminator."""

		#====================================== Training ===========================================#
		#===========================================================================================#
		
		#unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' %(self.model_type,self.num_epochs,self.lr,self.num_epochs_decay,self.augmentation_prob))

		unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' %(self.model_type,self.num_epochs,self.lr,self.batch_size,self.augmentation_prob))

		print (unet_path)

		# U-Net Train
		if os.path.isfile(unet_path):
			# Load the pretrained Encoder
			self.unet.load_state_dict(torch.load(unet_path))
			print('%s is Successfully Loaded from %s'%(self.model_type,unet_path))
			#=================================== Test ==============================================#
			print ('testing the model ...')

			self.unet.Conv1.register_forward_hook(get_activation('Conv1'))

			self.unet.train(False)
			self.unet.eval()


			acc = 0.	# Accuracy
			SE = 0.		# Sensitivity (Recall)
			SP = 0.		# Specificity
			PC = 0. 	# Precision
			F1 = 0.		# F1 Score
			JS = 0.		# Jaccard Similarity
			DC = 0.		# Dice Coefficient
			length=0
			for i, (images, GT, filename) in enumerate(self.valid_loader):

				images = images.to(self.device)
				print (images.size())
				#continue

				model = self.unet
				flops, params = profile(model, (images,))
				print (flops, params)			
				raise



				GT = GT.to(self.device)
				SR = F.sigmoid(self.unet(images))
				#SR = self.unet(images)

				SR = (SR > 0.5).type(torch.uint8)
				#GT = (GT > torch.max(GT)/2).type(torch.uint8)

				#TP = ((SR==1).type(torch.uint8)+(GT==1).type(torch.uint8))==2
				#FN = ((SR==0).type(torch.uint8)+(GT==1).type(torch.uint8))==2

				#print ((SR==1).data.cpu().numpy().max())
				#print ((GT==1).data.cpu().numpy().max())
				#print (TP.data.cpu().numpy().max())

				#SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)
				#print ('float(torch.sum(TP)): ', float(torch.sum(TP)))

				#print (type(SR.data.cpu().numpy()))
				#print(SR.data.cpu().numpy()[0,0,...].shape)

				#print (type(filename))
				#print (SR.data.cpu().numpy().shape)
				#raise

				img_save_path = os.path.join(self.result_path, 'res_vis')
				os.makedirs(img_save_path, exist_ok=True)

				#for i in range(len(filename)):
				#	#imsave(f'vis/skin/unet/{filename[0]}_GT.jpg', GT.data.cpu().numpy()[0,0,...]*255)				
				#	imsave(os.path.join(img_save_path, f'{filename[i]}_SR.jpg'), SR.data.cpu().numpy()[i,0,...]*255)
				#	#imsave(f'vis/skin/unet/{filename[0]}_images.jpg', images.data.cpu().numpy()[0,0,...])
				#continue

				act = activation['Conv1'].squeeze()
				print (act.size())
				print (act.max())
				print (act.min())
				act = (act-act.min())/(act.max()-act.min())
				print (act.max())
				print (act.min())
				raise
				row_n = 8
				for img_i in range(act.size(0)):
					fig, axarr = plt.subplots(row_n, act.size(1)//row_n, gridspec_kw = {'wspace':0.03, 'hspace':0.03})
					for idx_x in range(row_n):
						for idx_y in range(act.size(1)//row_n):
							axarr[idx_x, idx_y].imshow(act[img_i, idx_x*row_n+idx_y].cpu())
							axarr[idx_x, idx_y].axis('off')
					#plt.show()
					save_path = os.path.join(self.result_path, 'feature_img_train_Conv1')
					os.makedirs(save_path, exist_ok=True)
					f_name = filename[img_i]
					plt.savefig(os.path.join(save_path, f'{f_name}.jpg'))
					#print (filename)
					#print (self.result_path)
				#raise


				acc += get_accuracy(SR,GT)
				SE += get_sensitivity(SR,GT)
				SP += get_specificity(SR,GT)
				PC += get_precision(SR,GT)
				F1 += get_F1(SR,GT)
				JS += get_JS(SR,GT)
				DC += get_DC(SR,GT)
						
				#length += images.size(0)
				length += 1
					
			acc = acc/length
			SE = SE/length
			SP = SP/length
			PC = PC/length
			F1 = F1/length
			JS = JS/length
			DC = DC/length

			print('[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
					  acc,SE,SP,PC,F1,JS,DC))

			f = open(os.path.join(self.result_path,'result_test.csv'), 'a', encoding='utf-8', newline='')
			wr = csv.writer(f)
			wr.writerow(['model_type','acc','SE','SP','PC','F1','JS','DC'])
			wr.writerow([self.model_type,acc,SE,SP,PC,F1,JS,DC])
			f.close()
			

		else:

			print ('training the model ...')
			# Train for Encoder
			lr = self.lr
			#best_unet_score = 0.
			best_unet_score = 1000
			best_train_loss = 1000
			best_train_epoch = 0

			for epoch in range(self.num_epochs):

				self.unet.train(True)
				epoch_loss = 0
				
				acc = 0.	# Accuracy
				SE = 0.		# Sensitivity (Recall)
				SP = 0.		# Specificity
				PC = 0. 	# Precision
				F1 = 0.		# F1 Score
				JS = 0.		# Jaccard Similarity
				DC = 0.		# Dice Coefficient
				length = 0

				for i, (images, GT, filename) in enumerate(self.train_loader):
					# GT : Ground Truth
					#print (images.shape)
					#print (GT.shape)
					#raise

					#model = self.unet
					#flops, params = profile(model, (images,))
					#print (flops, params)			
					#raise


					images = images.to(self.device)
					GT = GT.to(self.device)

					# SR : Segmentation Result
					SR = self.unet(images)
					SR_probs = F.sigmoid(SR)
					SR_flat = SR_probs.view(SR_probs.size(0),-1)

					GT_flat = GT.view(GT.size(0),-1)
					loss = self.criterion(SR_flat,GT_flat)
					epoch_loss += loss.item()

					# Backprop + optimize
					self.reset_grad()
					loss.backward()
					self.optimizer.step()

					acc += get_accuracy(SR,GT)
					SE += get_sensitivity(SR,GT)
					SP += get_specificity(SR,GT)
					PC += get_precision(SR,GT)
					F1 += get_F1(SR,GT)
					JS += get_JS(SR,GT)
					DC += get_DC(SR,GT)

					#length += images.size(0)
					length += 1

				acc = acc/length
				SE = SE/length
				SP = SP/length
				PC = PC/length
				F1 = F1/length
				JS = JS/length
				DC = DC/length

				# Print the log info
				print('Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
					  epoch+1, self.num_epochs, \
					  epoch_loss,\
					  acc,SE,SP,PC,F1,JS,DC))

			

				# Decay learning rate
				#if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
					#lr -= (self.lr / float(self.num_epochs_decay))
					#for param_group in self.optimizer.param_groups:
						#param_group['lr'] = lr
					#print ('Decay learning rate to lr: {}.'.format(lr))

				if epoch_loss < best_train_loss:
					best_train_epoch = epoch

				if (epoch-best_train_epoch) > self.num_epochs_decay:
					best_train_epoch = epoch
					lr = self.lr * self.epochs_decay_rate
					for param_group in self.optimizer.param_groups:
						param_group['lr'] = lr
					print ('Decay learning rate to lr: {}.'.format(lr))				
				
				#===================================== Validation ====================================#
				self.unet.train(False)
				self.unet.eval()

				acc = 0.	# Accuracy
				SE = 0.		# Sensitivity (Recall)
				SP = 0.		# Specificity
				PC = 0. 	# Precision
				F1 = 0.		# F1 Score
				JS = 0.		# Jaccard Similarity
				DC = 0.		# Dice Coefficient
				length=0
				val_loss = 0

				for i, (images, GT, filename) in enumerate(self.test_loader):

					images = images.to(self.device)
					GT = GT.to(self.device)
					SR = F.sigmoid(self.unet(images))

					SR_flat = SR.view(SR.size(0),-1)

					GT_flat = GT.view(GT.size(0),-1)
					loss = self.criterion(SR_flat,GT_flat)
					val_loss += loss.item()

					acc += get_accuracy(SR,GT)
					SE += get_sensitivity(SR,GT)
					SP += get_specificity(SR,GT)
					PC += get_precision(SR,GT)
					F1 += get_F1(SR,GT)
					JS += get_JS(SR,GT)
					DC += get_DC(SR,GT)
						
					#length += images.size(0)
					length += 1					

				acc = acc/length
				SE = SE/length
				SP = SP/length
				PC = PC/length
				F1 = F1/length
				JS = JS/length
				DC = DC/length
				#unet_score = JS + DC
				unet_score = val_loss

				print('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f'%(acc,SE,SP,PC,F1,JS,DC))
				
				'''
				torchvision.utils.save_image(images.data.cpu(),
											os.path.join(self.result_path,
														'%s_valid_%d_image.png'%(self.model_type,epoch+1)))
				torchvision.utils.save_image(SR.data.cpu(),
											os.path.join(self.result_path,
														'%s_valid_%d_SR.png'%(self.model_type,epoch+1)))
				torchvision.utils.save_image(GT.data.cpu(),
											os.path.join(self.result_path,
														'%s_valid_%d_GT.png'%(self.model_type,epoch+1)))
				'''


				# Save Best U-Net model
				if unet_score < best_unet_score:
					best_unet_score = unet_score
					best_epoch = epoch
					best_unet = self.unet.state_dict()
					print('Best %s model score : %.4f'%(self.model_type,best_unet_score))
					torch.save(best_unet,unet_path)
					
			#===================================== test ====================================#
			del self.unet
			del best_unet
			self.build_model()
			self.unet.load_state_dict(torch.load(unet_path))
			
			self.unet.train(False)
			self.unet.eval()

			acc = 0.	# Accuracy
			SE = 0.		# Sensitivity (Recall)
			SP = 0.		# Specificity
			PC = 0. 	# Precision
			F1 = 0.		# F1 Score
			JS = 0.		# Jaccard Similarity
			DC = 0.		# Dice Coefficient
			length=0
			for i, (images, GT, filename) in enumerate(self.test_loader):

				images = images.to(self.device)
				GT = GT.to(self.device)
				SR = F.sigmoid(self.unet(images))
				acc += get_accuracy(SR,GT)
				SE += get_sensitivity(SR,GT)
				SP += get_specificity(SR,GT)
				PC += get_precision(SR,GT)
				F1 += get_F1(SR,GT)
				JS += get_JS(SR,GT)
				DC += get_DC(SR,GT)
						
				#length += images.size(0)
				length += 1					

			acc = acc/length
			SE = SE/length
			SP = SP/length
			PC = PC/length
			F1 = F1/length
			JS = JS/length
			DC = DC/length


			f = open(os.path.join(self.result_path,'result_val.csv'), 'a', encoding='utf-8', newline='')
			wr = csv.writer(f)
			wr.writerow(['model_type','acc','SE','SP','PC','F1','JS','DC','lr','best_epoch'])
			wr.writerow([self.model_type,acc,SE,SP,PC,F1,JS,DC,self.lr,best_epoch])
			f.close()
			














			
