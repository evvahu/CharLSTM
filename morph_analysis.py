import utils_data
import subprocess
import edit_morph

def get_morph_analysis(sent, apertium_mod='eus-morph'):
    bash_command = "echo \"{}\" | apertium {}".format(sent, apertium_mod)
   # process = subprocess.run(bash_command, text=True, shell=True, stdout=True)
    out=  subprocess.check_output(bash_command, shell=True, encoding='UTF-8')
    return out

if __name__ == "__main__":
    file_p = '/Users/eva/Documents/Work/experiments/Agent_first_project/agent_lms/stimuli/German2_psych_LSTM_nospill.csv'
    out_p = '/Users/eva/Documents/Work/experiments/Agent_first_project/agent_lms/stimuli/German2_psych_LSTM_nospill_morph2.csv'
    with open(out_p, 'w') as wf:
        wf.write('{}\t{}\t{}\t{}\n'.format('sent_id', 'cond', 'morph_analysis', 'word'))
        with open(file_p, 'r') as rf:
            next(rf)
            for l in rf:
                l = l.strip()
                line = l.split('\t')
                o = get_morph_analysis(line[2])
                morph = o.split(' ')
                whole_sent = []
                for i, morp in enumerate(morph):
                    fin_morp = ''
                    morp = morp.strip().strip('$')
                    if '/' in morp:
                        form, morp = morp.split('/', maxsplit=1)
                        if '/' in morp:
                            morp = morp.split('/')
                        else:
                            morp = [morp] 
                        morp = [m.replace('<', ' <') for m in morp]
            
                        if len(morp) > 1:
                            all_anns = []
                            for m in morp:
                                if (' ' in m) == True:
                                    print('init', m)
                                    l, t = m.split(' ', maxsplit=1)
                                    m_edited = edit_morph.edit_tags(m)
                                    if m_edited:
                                        all_anns.append(m_edited)
                                else:
                                    print('not init', m)
                                    all_anns.append(m)

                            merged = edit_morph.merge(all_anns)
                            fin_morp = ' '.join(merged)
                        else:
                            if morp[0].startswith('*'):
                                fin_morp = morp[0]
                            else:
                                if ' ' in morp[0]:
                                    m_edited = edit_morph.edit_tags(morp[0])
                                    fin_morp = ' '.join(m_edited)
                                else:
                                    fin_morp = morp[0]
                                
                    else:
                        fin_morp = morp
                    
                    whole_sent.append(fin_morp)
                    
                    
                sent_words = line[2].split(' ')
                #sent = ' '.join(whole_sent)
                for ws, wm in zip(sent_words, whole_sent):
                    wf.write('{}\t{}\t{}\t{}\n'.format(line[0], line[1], ws, wm))
                #wf.write('{}\t{}\t{}\n'.format(line[0], line[1], sent))

                    #print(morp)

                    
     
