
#path = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/GERMAN/fin_charmorph.txt'

def merge(anns):
    dict_merged = dict()
    div_wc = dict()
    fin_t = anns[0]
    not_fin = []
    for a in anns[1:]:
        if len(a) == len(fin_t):
            for i,l in enumerate(a):
                try:
                    if l == fin_t[i]:
                        continue
                    else:
                        old = fin_t[i]
                        if l not in fin_t[i]:
                            fin_t[i] = '{}/{}'.format(l, old)
                except:
                    continue

    return fin_t
    """ 
    for a in anns:
        if a[0] in dict_merged:
            dict_merged[a[0]].append(a)
        else:
            dict_merged[a[0]] = [a]
    dict_final = dict()
    for wc, tags in dict_merged.items():
        fin_t = tags[0]
        #for t in tags[1:]:
        for t in tags:
            if len(t) == len(fin_t): 
                for i,l in enumerate(t):
                    if l == fin_t[i]:
                        continue
                    else:
                        old = fin_t[i]
                        if l not in fin_t[i]:
                            fin_t[i] = '{}/{}'.format(l, old)
            else:
                for i, t in enumerate(tags[1:]):
                    dict_final['{}_{}'.format(wc, i)] = tags

        dict_final[wc] = fin_t
    
    if len(dict_final.keys()) > 1:   #@TODO: no solution yet 
        
    return ' '.join(['{}'.format(t) for t in list(dict_final.values())[0]])
    """  
def to_list(input):
    input = input.strip('[')
    input = input.strip(']')
    input = [i.strip() for i in input.split('\'') if (len(i) >0) and (i.strip() != ',')]
    return input

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

def edit_tags(tags):
    lemma, tags = tags.split(' ', maxsplit=1)
    tags = tags.split(' ')
    wclass = tags[0]
    plus = False
    for t in tags:
        if '+' in t:
            plus = True
    if plus:
        return _MW(tags)
    if wclass == '<n>':
        if 'lower' in tags:
            return 'NA'
        else:
            return tags
    elif wclass == '<adj>':
        if len(tags) >2:
            return tags[0:-2]
        else:
            return tags
    elif wclass.startswith('<v'):
        if 'heur' in tags:
            return 'NA'
        else:
            return tags
    elif wclass in ['<num>', '<prn>', '<cnjcoo>', '<ij>', '<cnjcoo>', '<abbr>', '<preadv>', '<cnsubj>','<cnjsub>','<cnjadv>', '<ito>', '<atp>', '<adv>', '<pr>', '<np>', '<cnadv>', '<cnadj>', '<pprep>', '<det>']: #cnsubj
        return tags
    #else:
       # print('not in here', wclass)
"""
i = 0
o = 0
os = list()
out_p = '/Users/eva/Documents/Work/experiments/Agent_first_project/Surprisal_LMs/data/GERMAN/out_morphs_sents.txt' 
writer = open(out_p, 'w') 
with open(path, 'r') as rf:
    for i, l in enumerate(rf):
        l = l.strip()
        if not l: continue
        line = l.split('\t')
        try:
            morphs = to_list(line[1])
            fin_annotation = str()
            if len(morphs) > 1:
                all_anns = []
                for m in morphs:
                    try:
                        if (not m.startswith('*')) and ('<sent>' not in m):
                            out = edit_tags(m)
                            if out:
                                all_anns.append(out)
                        if '<sent>' in m:
                            m = m.strip('$').strip('^')
                            all_anns.append(m)
                    except:
                        all_anns.append('<unk>')
                        #print('here', m)

                if len(all_anns) > 0:
                    if 'sent' in all_anns[0]:
                        fin_annotation = '{} <sent>'.format(all_anns[0][0])
                    elif len(all_anns) > 1:
                        merged = merge(all_anns)
                        fin_annotation = ' '.join(merged)
                    else:
                        fin_annotation = ' '.join(all_anns[0])
            fin_annotation = fin_annotation.strip('^C')
            fin_annotation = fin_annotation.replace('  ', '')
            fin_annotation = fin_annotation.strip()
            writer.write(' {}'.format(fin_annotation))
            if '<sent>' in fin_annotation:
                writer.write('\n')
        except:
            print(line)
            #    merged = merge(all_anns)
            #for m in merged:
            #    writer.write(m)
                #print(out)


        #except:
        #print(morphs)

"""