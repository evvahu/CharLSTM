import re
import random

def merge(anns):
    print(anns)
    dict_merged = dict()
    for a in anns:
        if a[0] in dict_merged:
            dict_merged[a[0]].append(a)
        else:
            dict_merged[a[0]] = [a]
    dict_final = dict()
    for wc, tags in dict_merged.items():
        fin_t = tags[0]
      
        for t in tags:
            for i, l in enumerate(t):
                if l == fin_t[i]:
                    continue
                else:
                    old = fin_t[i]
                    if l not in fin_t[i]:
                        fin_t[i] = '{}/{}'.format(l, old)
        dict_final[wc] = fin_t
    if len(dict_final.keys()) > 1:   #@TODO: no solution yet 
        print('dict_final', dict_final)
    print(dict_final)
    return ' '.join(['<{}>'.format(t) for t in list(dict_final.values())[0]])
            

def edit_tags(tags):
  
    tag_type = tags[0]
    plus = False
    for t in tags:
        if '+' in t:
            plus = True
    if plus:
        return _MW(tags)
    else:
        if tag_type == 'n': #or np?
            return _n(tags)
        elif tag_type.startswith('v'):
            return _v(tags)
        elif tag_type in ['num', 'prn', 'cnjcoo', 'ij', 'cnjcoo', 'abbr', 'preadv', 'cnsubj','cnjsub','cnjadv', 'ito', 'atp', 'adv', 'pr', 'np', 'cnadv', 'cnadj', 'pprep', 'det']: #cnsubj
            return tags
        elif tag_type == 'adj':
            return _adj(tags)
        else:
            return ''
# 'prn', 'ind': delete end 
# vblex: sep, inf, ger

def _n(tags):
    comp = False
    for t in tags:
        if '+' in t:
            comp = True
    if comp:
        _MW(tags)
    else:
        if 'lower' in tags:
            return 'NA'
        else:
            return tags

def _MW(tags):
    comp = tags[0] == 'n'
    tags = '**'.join(tags)
    tags_cspl = tags.split('+')
    if 'lower' in tags_cspl[0]:
        return 'NA'
    else:
        fin_tags = tags_cspl[0].split('**')
        if comp:
            if 'lower' in fin_tags:
                return 'NA'
        for mc in tags_cspl[1:]:
            mc = mc.split('**')
            fin_tags.append('+')
            for tag in mc:
                tag = tag.strip('+')
                if tag == 'lower':
                    continue
                else:
                    fin_tags.append(tag)

    return [t for t in fin_tags if len(t) > 0]

def _v(tags):
        # delete 'heur'
    if 'heur' in tags:
        return 'NA'
    return tags#[tags[0]] + tags[2:]

def _adj(tags):
    if len(tags) > 2:
        return tags[0:-2]
    else:
        return tags
    
#mfn = "m/f/n"
# prn, det, n, vblex, np, adj, cnsubj, pr, pprep, adv, vbmod, vaux, num, vbser, cnjcoo, preadv
# prpers 
# 
#  
if __name__ == "__main__":
    path = '/Users/eva/apertium/try_basque_wiki.txt'
    oov = 0
    iv = 0
    oov_l = []
    voc_set = set()
    writer = open('/Users/eva/apertium/out_basque.txt', 'w')
    with open(path, 'r') as rf:
        for l in rf:
            l = l.strip().strip('q^')
            line = l.split('^')
            sent = ''
            for w in line:
                try:
                    fin_ann = ''
                    _, ann = w.split('/', maxsplit=1)
                    if ann.startswith('*'):
                        oov +=1
                        oov_l.append(ann)
                        fin_ann = ' <unk>'
                    else:
                        iv +=1
                        if '/' in ann:
                            #lemm, ann = ann.strip('$').split('<', maxsplit=1)
                            #ann = '<' + ann
                            ann = ann.strip(' ').strip('$')
                            ann_spl = ann.split('/')
                            anns = []
                            for a in ann_spl:
                                print(a)
                                a_spl = re.split('<|>', a)
                                a_spl = [t for t in a_spl if len(t) > 0]
                                a_spl = [t.strip('>') for t in a_spl]
                                a_spl_edited = edit_tags(a_spl[1:])
                                if len(a_spl_edited) < 1:
                                    print(a_spl, a_spl_edited)
                                if a_spl_edited != 'NA' and len(a_spl_edited) > 1:
                                    anns.append(a_spl_edited)

                            fin_ann = ' {} {} '.format(a_spl[0],merge(anns))
                            
                        else:
                            if '<sent>' in ann:
                                fin_ann = ' {} '.format(ann[0])
                            else:
                                fin_ann = ann.strip(' ').strip('$').replace('<', ' <')
                   
                    sent += fin_ann
                except:
                    continue
            sent = sent.replace('<sent>', '')
            sent = sent.replace('>.', '>')
            sent = sent.replace(' .  . ', ' .')
            sent = sent.replace('  ', ' ')
            writer.write('{}\n'.format(sent))
    writer.close()
    #print(len(voc_set), voc_set)


# + in compounds 