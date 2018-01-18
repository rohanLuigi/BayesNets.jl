"""
Exact inference using factors and variable eliminations
"""
struct ExactInference <: InferenceMethod
end
struct GreedyExactInference <: InferenceMethod
 end

function infer{BN<:DiscreteBayesNet}(im::ExactInference, inf::InferenceState{BN})

    bn = inf.pgm
    nodes = names(bn)
    query = inf.query
    evidence = inf.evidence
    hidden = setdiff(nodes, vcat(query, names(evidence)))
    factors = map(n -> Factor(bn, n, evidence), nodes)
    # successively remove the hidden nodes
    # order impacts performance, but no ordering heuristics used here
    return removeNodes(hidden,factors)
end

function removeNodes(hidden,factors) 
    for h in hidden
        # find the facts that contain the hidden variable
        contain_h = filter(ϕ -> h in ϕ, factors)
        # add the product of those factors to the set
        if !isempty(contain_h)
            # remove those factors
            factors = setdiff(factors, contain_h)
            push!(factors, sum(reduce((*), contain_h), h))
        end
    end
    ϕ = normalize!(reduce((*), factors))
    return ϕ
end

function getCutsetFor(i,cutsets)
    contain_i = filter(ϕ -> i in ϕ, cutsets)
    s=Set(Symbol[])
    for j in contain_i
        union!(s,j)
    end
    return s
end

function greedyOrder(nodes,factors)
    cutsets = [Set(i.dimensions) for i in factors]
    for i=1:length(nodes)
        minLength = 100000
        minj = 1
        minc = Set()
        for j=i:length(nodes)
            cj = getCutsetFor(nodes[j],cutsets)
            lj = length(cj)
            if lj < minLength
                minLength = lj
                minj = j
                minc = copy(cj)
            end
        end
        nodes[i],nodes[minj] = nodes[minj],nodes[i]
        contain_min = filter(ϕ -> nodes[i] in ϕ, cutsets)
        cutsets     = setdiff(cutsets,contain_min)
        minc        = setdiff(minc,[nodes[i]])
        push!(cutsets,Set(minc))
    end
    cutsets
end

function infer{BN<:DiscreteBayesNet}(im::GreedyExactInference, inf::InferenceState{BN})
    bn = inf.pgm
    nodes = names(bn)
    query = inf.query
    evidence = inf.evidence
    hidden = setdiff(nodes, vcat(query, names(evidence)))
    factors = map(n -> Factor(bn, n, evidence), nodes)
    # successively remove the hidden nodes
    # order impacts performance, and greedy ordering is used
    greedyOrder(hidden,factors)
    return removeNodes(hidden,factors)

end

infer{BN<:DiscreteBayesNet}(inf::InferenceState{BN}) = infer(ExactInference(), inf)
infer{BN<:DiscreteBayesNet}(bn::BN, query::NodeNameUnion; evidence::Assignment=Assignment()) = infer(ExactInference(), InferenceState(bn, query, evidence))
infer{BN<:DiscreteBayesNet}(im::GreedyExactInference,bn::BN, query::NodeNameUnion; evidence::Assignment=Assignment()) = infer(im, InferenceState(bn, query, evidence))

