from apiq.quant_linear import QuantLinear


def quantize_llama_like(model, weight_quant_params):
    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP
    from transformers.models.mistral.modeling_mistral import MistralAttention, MistralMLP
    
    for _, m in model.model.named_modules():
        if isinstance(m, (LlamaMLP, MistralMLP)):
            try:
                m.gate_proj.base_layer = QuantLinear(m.gate_proj.base_layer, weight_quant_params = weight_quant_params)
            except:
                m.gate_proj = QuantLinear(m.gate_proj, weight_quant_params = weight_quant_params)
                
            try:
                m.up_proj.base_layer = QuantLinear(m.up_proj.base_layer, weight_quant_params = weight_quant_params)
            except:
                m.up_proj = QuantLinear(m.up_proj, weight_quant_params = weight_quant_params)
                
            try:
                m.down_proj.base_layer = QuantLinear(m.down_proj.base_layer, weight_quant_params = weight_quant_params)
            except:
                m.down_proj = QuantLinear(m.down_proj, weight_quant_params = weight_quant_params)
                
        elif isinstance(m, (LlamaAttention, MistralAttention)):
            try:
                m.q_proj.base_layer = QuantLinear(m.q_proj.base_layer, weight_quant_params = weight_quant_params)
            except:
                m.q_proj = QuantLinear(m.q_proj, weight_quant_params = weight_quant_params)
                
            try:
                m.k_proj.base_layer = QuantLinear(m.k_proj.base_layer, weight_quant_params = weight_quant_params)
            except:
                m.k_proj = QuantLinear(m.k_proj, weight_quant_params = weight_quant_params)
                
            try:
                m.v_proj.base_layer = QuantLinear(m.v_proj.base_layer, weight_quant_params = weight_quant_params)
            except:
                m.v_proj = QuantLinear(m.v_proj, weight_quant_params = weight_quant_params)
                
            try:
                m.o_proj.base_layer = QuantLinear(m.o_proj.base_layer, weight_quant_params = weight_quant_params)
            except:
                m.o_proj = QuantLinear(m.o_proj, weight_quant_params = weight_quant_params)
                
    return model