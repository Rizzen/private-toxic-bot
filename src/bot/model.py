import model_utils as mdu

class Model():
    def __init__(self, checkpoint: str):
        (self.model, self.tokenizer, self.memory) = mdu.init_model(checkpoint)
        self.memory_size = 511
        self.messages_history = []

    def process_message(self, message_text):
        self.messages_history.append(message_text)

    def predict(self, text):
        input = text.lower()
        (answer, new_history) = mdu.predict_answer(input, self.model, self.tokenizer, self.memory, self.memory_size)
        self.memory = new_history
        print('---------------------------------------------------------')
        print('Input: ' + text)
        print('----------------------------------------------------------')
        print('Out: ' + answer)
        print('-------------------------HISTORY--------------------------')
        print(self.memory.shape[1])
        print('----------------------------------------------------------')
        self.adjust_memory(self.memory)
        return answer

    def predict2(self, text):
        answer = mdu.predict2(self.model, self.tokenizer, self.messages_history)
        print('---------------------------------------------------------')
        print('Input: ' + text)
        print('----------------------------------------------------------')
        print('Out: ' + answer)
        return answer

    def adjust_memory(self, new_history):
        if new_history.shape[1] >= self.memory_size:
            new_memory = new_history[:, :self.memory_size]
            print("adjusted mem: " + str(new_memory.shape[1]))
            self.memory = new_memory

    @staticmethod
    def adjust_seq_size(tensor, max_size):
        if tensor.shape[1] >= max_size:
            result = tensor[:, :max_size]
            return result
        return tensor