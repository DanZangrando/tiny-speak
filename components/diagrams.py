import graphviz

def get_common_node_style():
    return 'fontname="Helvetica", shape=box, style="filled,rounded", fontcolor="white"'

def get_common_edge_style():
    return 'fontname="Helvetica", color="#adb5bd", fontcolor="#adb5bd"'

def get_recognizer_diagram():
    dot = graphviz.Digraph('TinyRecognizer', comment='TinyRecognizer Architecture')
    dot.attr(rankdir='TB', bgcolor='#0e1117', fontname='Helvetica', fontcolor='white')
    
    # Styles
    dot.attr('node', **{'fontname': 'Helvetica', 'shape': 'box', 'style': 'filled,rounded', 'fontcolor': 'white', 'color': '#303030'})
    dot.attr('edge', **{'fontname': 'Helvetica', 'color': '#6c757d', 'fontcolor': '#adb5bd'})

    # Input
    dot.node('Input', 'Input Image\n(3 x 64 x 64)\nRGB', fillcolor='#1e88e5', color='#1e88e5')

    # Backbone
    with dot.subgraph(name='cluster_backbone') as c:
        c.attr(label='CORnet-Z Backbone (Visual Cortex)', color='#6c757d', fontcolor='white', style='dashed')
        
        c.node('V1', 'V1 Block\nConv2d(3→64, k=7, s=2)\nReLU, MaxPool', fillcolor='#263238')
        c.node('V2', 'V2 Block\nConv2d(64→128, k=3)\nReLU, MaxPool', fillcolor='#263238')
        c.node('V4', 'V4 Block\nConv2d(128→256, k=3)\nReLU, MaxPool', fillcolor='#263238')
        c.node('IT', 'IT Block\nConv2d(256→512, k=3)\nReLU, MaxPool', fillcolor='#263238')

    # Decoder
    with dot.subgraph(name='cluster_decoder') as c:
        c.attr(label='Decoder / Classifier', color='#6c757d', fontcolor='white', style='dashed')
        
        c.node('AvgPool', 'AdaptiveAvgPool2d\n(1x1)', fillcolor='#424242')
        c.node('Flatten', 'Flatten\n(Batch, 512)', fillcolor='#424242')
        c.node('Linear1', 'Linear(512 → 1024)\nReLU', fillcolor='#d81b60') # Pinkish for learnable
        c.node('Linear2', 'Linear(1024 → 768)\n[Wav2Vec Dim]', fillcolor='#d81b60')
        c.node('Classifier', 'Linear(768 → NumClasses)', fillcolor='#d81b60')

    # Output
    dot.node('Output', 'Logits / Probabilities\n(Batch, NumClasses)', fillcolor='#43a047', color='#43a047')

    # Edges
    dot.edge('Input', 'V1')
    dot.edge('V1', 'V2')
    dot.edge('V2', 'V4')
    dot.edge('V4', 'IT')
    dot.edge('IT', 'AvgPool')
    dot.edge('AvgPool', 'Flatten')
    dot.edge('Flatten', 'Linear1')
    dot.edge('Linear1', 'Linear2')
    dot.edge('Linear2', 'Classifier')
    dot.edge('Classifier', 'Output')

    return dot

def get_listener_diagram():
    dot = graphviz.Digraph('TinyListener', comment='TinyListener Architecture')
    dot.attr(rankdir='TB', bgcolor='#0e1117', fontname='Helvetica', fontcolor='white')
    
    dot.attr('node', **{'fontname': 'Helvetica', 'shape': 'box', 'style': 'filled,rounded', 'fontcolor': 'white', 'color': '#303030'})
    dot.attr('edge', **{'fontname': 'Helvetica', 'color': '#6c757d', 'fontcolor': '#adb5bd'})

    dot.node('Input', 'Audio Waveform\n(Batch, Samples)', fillcolor='#1e88e5', color='#1e88e5')

    with dot.subgraph(name='cluster_wav2vec') as c:
        c.attr(label='Wav2Vec 2.0 (Frozen)', color='#6c757d', fontcolor='white', style='filled', fillcolor='#212529')
        c.node('FeatExt', 'Feature Extractor\n(Conv Layers)', fillcolor='#37474f')
        c.node('Transformer', 'Transformer Encoder\n(Self-Attention)', fillcolor='#37474f')
        c.node('LayerSel', 'Layer Selection\n(Layer 5)', fillcolor='#37474f')

    dot.node('Process', 'Mask & Downsample\n(Factor 7)', fillcolor='#ff6f00')

    with dot.subgraph(name='cluster_tinyspeak') as c:
        c.attr(label='TinySpeak (Learnable)', color='#6c757d', fontcolor='white', style='dashed')
        c.node('LSTM', 'LSTM Encoder\n(Input: 768, Hidden: 128)', fillcolor='#d81b60')
        c.node('Head', 'Classification Head\nLinear(128 → NumClasses)', fillcolor='#d81b60')

    dot.node('Output', 'Logits\n(Batch, NumClasses)', fillcolor='#43a047', color='#43a047')

    dot.edge('Input', 'FeatExt')
    dot.edge('FeatExt', 'Transformer')
    dot.edge('Transformer', 'LayerSel')
    dot.edge('LayerSel', 'Process')
    dot.edge('Process', 'LSTM')
    dot.edge('LSTM', 'Head')
    dot.edge('Head', 'Output')

    return dot

def get_reader_diagram():
    dot = graphviz.Digraph('TinyReader', comment='TinyReader Architecture')
    dot.attr(rankdir='TB', bgcolor='#0e1117', fontname='Helvetica', fontcolor='white')
    
    dot.attr('node', **{'fontname': 'Helvetica', 'shape': 'box', 'style': 'filled,rounded', 'fontcolor': 'white', 'color': '#303030'})
    dot.attr('edge', **{'fontname': 'Helvetica', 'color': '#6c757d', 'fontcolor': '#adb5bd'})

    # Inputs
    dot.node('Input', 'Concept Input\n(Logits / One-Hot)', fillcolor='#1e88e5', color='#1e88e5')

    # Generator
    with dot.subgraph(name='cluster_generator') as c:
        c.attr(label='TinyReader (Generator)', color='#6c757d', fontcolor='white', style='dashed')
        c.node('Encoder', 'Concept Encoder\nLinear(NumClasses → Hidden)', fillcolor='#d81b60')
        c.node('LSTM', 'LSTM Decoder\n(Unroll T steps)', fillcolor='#d81b60')
        c.node('Proj', 'Output Projection\nLinear(Hidden → 768)', fillcolor='#d81b60')

    dot.node('Generated', 'Generated Embeddings\n(Batch, Time, 768)', fillcolor='#00acc1', shape='parallelogram')

    # Loss Components
    with dot.subgraph(name='cluster_loss') as c:
        c.attr(label='Hybrid Loss System', color='#6c757d', fontcolor='white', style='dotted')
        
        c.node('Real', 'Real Embeddings\n(Ground Truth)', fillcolor='#546e7a', shape='parallelogram')
        
        c.node('MSE', 'MSE Loss\n(Reconstruction)', fillcolor='#e53935', shape='ellipse')
        c.node('Cos', 'Cosine Loss\n(Structure)', fillcolor='#e53935', shape='ellipse')
        
        with c.subgraph(name='cluster_innerear') as ie:
            ie.attr(label='Inner Ear (Frozen TinyListener)', color='#6c757d', style='filled', fillcolor='#212529')
            ie.node('Listener', 'TinySpeak\n(LSTM + Head)', fillcolor='#37474f')
            ie.node('Pred', 'Predicted Logits', fillcolor='#37474f')
            
        c.node('Perceptual', 'Perceptual Loss\n(CrossEntropy)', fillcolor='#e53935', shape='ellipse')

    # Edges
    dot.edge('Input', 'Encoder')
    dot.edge('Encoder', 'LSTM')
    dot.edge('LSTM', 'Proj')
    dot.edge('Proj', 'Generated')

    dot.edge('Generated', 'MSE')
    dot.edge('Real', 'MSE')

    dot.edge('Generated', 'Cos')
    dot.edge('Real', 'Cos')

    dot.edge('Generated', 'Listener')
    dot.edge('Listener', 'Pred')
    dot.edge('Pred', 'Perceptual')
    dot.edge('Input', 'Perceptual', label='Target', style='dotted')

    return dot
