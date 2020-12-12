import numpy as np

class ReplayBuffer():

	def  __init__(self, memory_size : int , batch_size : int = 32):

		self.image_size_w = 224
		self.image_size_h = 224
		self.channel = 4
        	self.obs_buf = np.zeros([memory_size, self.channel, self.image_size_h, self.image_size_w], dtype=np.float32)
        	self.next_obs_buf = np.zeros([memory_size , self.channel, self.image_size_h, self.image_size_w], dtype=np.float32)
    		self.act_buf = np.zeros([memory_size], dtype = np.float32)
    		self.rew_buf = np.zeros([memory_size], dtype = np.float32)        	
        	self.done_buf = np.zeros(memory_size, dtype = np.float32)
        
    		self.max_size, self.batch_size = memory_size, batch_size
    		self.ptr, self.size,  = 0, 0 


	def store(self, obs : np.ndarray, 
                    	act : np.ndarray,
                	rew : float,
                	next_obs : np.ndarray,
                	done : bool):
        
        	self.obs_buf[self.ptr] = obs
        	self.next_obs_buf[self.ptr] = next_obs
        	self.act_buf[self.ptr] = act
        	self.rew_buf[self.ptr] = rew
        	self.done_buf[self.ptr] = done
        
        	self.ptr = (self.ptr + 1) % self.max_size   
        	self.size = min(self.size + 1, self.max_size) 
   

	def sample_batch(self):
        
		idxs = np.random.choice(self.size , self.batch_size , replace = False)
        
		return dict(obs = self.obs_buf[idxs],
                    	    act = self.act_buf[idxs],
                	    rew = self.rew_buf[idxs],
                	    next_obs = self.next_obs_buf[idxs],
                	    done = self.done_buf[idxs])

	def __len__(self):
        
		return self.size
