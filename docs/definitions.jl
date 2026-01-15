""" 
    convert_newcommand_to_pair(s::AbstractString)

Converts a string of LaTeX to a pair, e.g. \\newcommand{\\RR}{\\mathbb{R}} to :RR => "\\mathbb{R}".
"""
function convert_newcommand_to_pair(s::AbstractString)
    if startswith(s, "\\newcommand{\\")
        parts = split(s, "}{")
        @assert length(parts) == 2
        cmd = replace(parts[1], "\\newcommand{\\" => "")
        return Symbol(cmd) => parts[2][1:end-1]
    end

    return nothing
end

"""
    make_macros_dict(definitions_path::AbstractString)

Create a `Documenter.HTMLWriter.MathJax3` `config` based on a LaTeX file containing `\\newcomand` definitions.
"""
function make_macros_dict(definitions_path::AbstractString)
    defs_txt = read(definitions_path, String)
    pairs = filter(
        !isnothing,
        map(convert_newcommand_to_pair ∘ strip, split(defs_txt, "\n"))
    )
    return Dict(pairs)
end