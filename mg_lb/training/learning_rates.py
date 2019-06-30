from word2number import w2n


class half_every_n:
    def __init__(self, n):
        self.n = n

    def __call__(self, start_lr, epoch):
        i_decay = epoch // self.n
        lr = start_lr / (2 ** i_decay)
        return lr


def get_decay(name):
    if name[0:10] == 'half_every':
        num = w2n.word_to_num(name[11:])
        dec = half_every_n(num)
    return dec