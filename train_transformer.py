from preprocess import get_dataset, DataLoader, collate_fn_transformer
from network import *
from plotting_utils import plot_alignment_to_numpy
from tensorboardX import SummaryWriter
import wandb
import argparse
import torchvision.utils as vutils
import os
from tqdm import tqdm

def adjust_learning_rate(optimizer, step_num, warmup_step=4000):
    lr = hp.lr * warmup_step**0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str,
                        help='give a distinct name for this running')
    parser.add_argument('--prj_name', type=str, default='tts-tacotron2',
                        help='give a project name for this running')
    args = parser.parse_args()

    dataset = get_dataset()
    global_step = 0

    m = nn.DataParallel(Model().cuda())

    m.train()
    optimizer = t.optim.Adam(m.parameters(), lr=hp.lr)

    pos_weight = t.FloatTensor([5.]).cuda()
    writer = SummaryWriter()
    wandb.init(name=args.run_name, project=args.prj_name)

    for epoch in range(hp.epochs):

        dataloader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True, collate_fn=collate_fn_transformer, drop_last=True, num_workers=16)
        pbar = tqdm(dataloader)
        for i, data in enumerate(pbar):
            pbar.set_description("Processing at epoch %d"%epoch)

            character, mel, mel_input, pos_text, pos_mel, _ = data

            stop_tokens = t.abs(pos_mel.ne(0).type(t.float) - 1)

            character = character.cuda()
            mel = mel.cuda()
            mel_input = mel_input.cuda()
            pos_text = pos_text.cuda()
            pos_mel = pos_mel.cuda()

            mel_pred, postnet_pred, attn_probs, stop_preds, attns_enc, attns_dec = m.forward(character, mel_input, pos_text, pos_mel)

            mel_loss = nn.L1Loss()(mel_pred, mel)
            post_mel_loss = nn.L1Loss()(postnet_pred, mel)

            loss = mel_loss + post_mel_loss

            optimizer.zero_grad()
            # Calculate gradients
            loss.backward()

            nn.utils.clip_grad_norm_(m.parameters(), 1.)

            # Update weights
            optimizer.step()

            # Adapt learning sample_rate
            global_step += 1
            if global_step < 400000:
                adjust_learning_rate(optimizer, global_step)

            # Logging ##########################################################

            #writer.add_scalars('training_loss',{
            #        'mel_loss':mel_loss,
            #        'post_mel_loss':post_mel_loss,

            #    }, global_step)
            #writer.add_scalars('alphas',{
            #        'encoder_alpha':m.module.encoder.alpha.data,
            #        'decoder_alpha':m.module.decoder.alpha.data,
            #    }, global_step)

            wandb.log({"epoch": epoch,
                       "iteration":global_step,
                       "training.mel_loss": mel_loss,
                       "training.post_mel_loss": post_mel_loss,
                       "training.loss":loss,
                       "training.encoder_alpha": m.module.encoder.alpha.data.cpu().numpy(),
                       "training.decoder_alpha": m.module.decoder.alpha.data.cpu().numpy()
                       })

            hp.image_step = len(dataloader)
            # If this iteration is last in this epoch, then ...
            if (global_step % hp.image_step) == (hp.image_step - 1) or (global_step == 1):
                attn_img_list_old = list()
                attn_img_list = list()
                attn_enc_img_list_old = list()
                attn_enc_img_list = list()
                attn_dec_img_list_old = list()
                attn_dec_img_list = list()


                for i, prob in enumerate(attn_probs):

                    num_h = prob.size(0) # num_h == 3
                    for j in range(4):

                        x1 = vutils.make_grid(prob[j*16] * 255)
                        x2 = vutils.make_grid(prob[j*16])

                        #writer.add_image('Attention_%d_0'%global_step, x, i*4+j)

                        #wandb.log({"Attention": [wandb.Image(plot_alignment_to_numpy(x.data.cpu().numpy().T, caption='Attention_%d_0'%global_step)]},
                        #          step=i*4+j)
                        #wandb.log({"Attention": wandb.Image(x.data.cpu().numpy().T, caption='Attention_%d_0'%global_step)}, step=i*4+j)
                        #wandb.log({"Attention": [wandb.Image(x.data.cpu().numpy().T, caption='Attention_{}_{}'.format(epoch+1, global_step))]})
                        #wandb.log({"Attention": [wandb.Image(plot_alignment_to_numpy(x.data.cpu().numpy().T), caption='Attention_{}_{}'.format(epoch+1, global_step))]})
                        '''wandb.log({"Attention": [
                            wandb.Image(x1.data.cpu().numpy().T, caption='Attention_{}_{}'.format(epoch+1, global_step)),
                            wandb.Image(plot_alignment_to_numpy(x2.data.cpu().numpy().T), caption='Attention_{}_{}'.format(epoch+1, global_step))
                        ]})'''

                        attn_img_list_old.append(wandb.Image(x1.data.cpu().numpy().T, caption='Attention_{}_{}'.format(epoch+1, global_step)))
                        attn_img_list.append(wandb.Image(plot_alignment_to_numpy(x2.data.cpu().numpy().T), caption='Attention_{}_{}'.format(epoch+1, global_step)))

                for i, prob in enumerate(attns_enc):

                    num_h = prob.size(0)
                    for j in range(4):

                        x1 = vutils.make_grid(prob[j*16] * 255)
                        x2 = vutils.make_grid(prob[j*16])

                        #writer.add_image('Attention_enc_%d_0'%global_step, x, i*4+j)
                        #wandb.log({"Attention_enc": [wandb.Image(plot_alignment_to_numpy(x.data.cpu().numpy().T, caption='Attention_enc_%d_0'%global_step)]})
                        #wandb.log({"Attention_enc": [wandb.Image(x.data.cpu().numpy().T, caption='Attention_enc_%d_0'%global_step)]}, step=i*4+j)
                        #wandb.log({"Attention_enc": [wandb.Image(x.data.cpu().numpy().T, caption='Attention_enc_{}_{}'.format(epoch+1, global_step))]})
                        attn_enc_img_list_old.append(wandb.Image(x1.data.cpu().numpy().T, caption='Attention_enc_{}_{}'.format(epoch+1, global_step)))
                        attn_enc_img_list.append(wandb.Image(plot_alignment_to_numpy(x2.data.cpu().numpy().T), caption='Attention_enc_{}_{}'.format(epoch+1, global_step)))

                for i, prob in enumerate(attns_dec):

                    num_h = prob.size(0)
                    for j in range(4):

                        x1 = vutils.make_grid(prob[j*16] * 255)
                        x2 = vutils.make_grid(prob[j*16])

                        #writer.add_image('Attention_dec_%d_0'%global_step, x, i*4+j)
                        #wandb.log({"Attention_dec": [wandb.Image(plot_alignment_to_numpy(x.data.cpu().numpy().T, caption='Attention_dec_%d_0'%global_step)]})
                        #wandb.log({"Attention_dec": [wandb.Image(x.data.cpu().numpy().T, caption='Attention_dec_%d_0'%global_step)]}, step=i*4+j)
                        #wandb.log({"Attention_dec": [wandb.Image(x.data.cpu().numpy().T, caption='Attention_dec_{}_{}'.format(epoch+1, global_step))]})
                        attn_dec_img_list_old.append(wandb.Image(x1.data.cpu().numpy().T, caption='Attention_dec_{}_{}'.format(epoch+1, global_step)))
                        attn_dec_img_list.append(wandb.Image(plot_alignment_to_numpy(x2.data.cpu().numpy().T), caption='Attention_dec_{}_{}'.format(epoch+1, global_step)))

                # wandb attention map logging
                wandb.log({
                    "Attention_old":attn_img_list_old,
                    "Attention": attn_img_list,
                    "Attention_enc_old":attn_enc_img_list_old,
                    "Attention_enc":attn_enc_img_list,
                    "Attention_dec_old":attn_dec_img_list_old,
                    "Attention_dec":attn_dec_img_list
                })

            hp.save_step = 5 * len(dataloader)
            if global_step % hp.save_step == 0:
                t.save({'model':m.state_dict(),
                        'optimizer':optimizer.state_dict()},
                    os.path.join(hp.checkpoint_path,'checkpoint_transformer_{}.pth.tar'.format(global_step)))


if __name__ == '__main__':
    main()
