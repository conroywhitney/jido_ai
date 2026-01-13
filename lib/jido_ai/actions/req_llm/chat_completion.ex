defmodule Jido.AI.Actions.ReqLlm.ChatCompletion do
  @moduledoc """
  Chat completion action using ReqLLM for multi-provider support.

  This action provides direct access to chat completion functionality across
  57+ providers through ReqLLM, replacing the LangChain-based implementation
  with lighter dependencies and broader provider support.

  ## Features

  - Multi-provider support (57+ providers via ReqLLM)
  - Tool/function calling capabilities
  - Response quality control with retry mechanisms
  - Support for various LLM parameters (temperature, top_p, etc.)
  - Structured error handling and logging
  - Streaming support (when provider allows)

  ## Usage

  ```elixir
  # Basic usage
  {:ok, result} = Jido.AI.Actions.ReqLlm.ChatCompletion.run(%{
    model: %Jido.AI.Model{provider: :anthropic, model: "claude-3-sonnet-20240229"},
    prompt: Jido.AI.Prompt.new(:user, "What's the weather in Tokyo?")
  })

  # With function calling / tools
  {:ok, result} = Jido.AI.Actions.ReqLlm.ChatCompletion.run(%{
    model: %Jido.AI.Model{provider: :openai, model: "gpt-4o"},
    prompt: prompt,
    tools: [Jido.Actions.Weather.GetWeather, Jido.Actions.Search.WebSearch],
    temperature: 0.2
  })

  # Streaming responses
  {:ok, stream} = Jido.AI.Actions.ReqLlm.ChatCompletion.run(%{
    model: model,
    prompt: prompt,
    stream: true
  })

  Enum.each(stream, fn chunk ->
    IO.puts(chunk.content)
  end)
  ```

  ## Support Matrix

  Supports all providers available in ReqLLM (57+), including:
  - OpenAI (GPT models)
  - Anthropic (Claude models)
  - Google (Gemini models)
  - Mistral, Cohere, Groq, and many more

  See ReqLLM documentation for full provider list.
  """
  use Jido.Action,
    name: "reqllm_chat_completion",
    description: "Chat completion action using ReqLLM",
    schema: [
      model: [
        type: {:custom, Jido.AI.Model, :validate_model_opts, []},
        required: true,
        doc:
          "The AI model to use (e.g., {:anthropic, [model: \"claude-3-sonnet-20240229\"]} or %Jido.AI.Model{})"
      ],
      prompt: [
        type: {:custom, Jido.AI.Prompt, :validate_prompt_opts, []},
        required: true,
        doc: "The prompt to use for the response"
      ],
      tools: [
        type: {:list, :atom},
        required: false,
        doc: "List of Jido.Action modules for function calling"
      ],
      max_retries: [
        type: :integer,
        default: 0,
        doc: "Number of retries for validation failures"
      ],
      temperature: [type: :float, default: 0.7, doc: "Temperature for response randomness"],
      max_tokens: [type: :integer, default: 1000, doc: "Maximum tokens in response"],
      top_p: [type: :float, doc: "Top p sampling parameter"],
      stop: [type: {:list, :string}, doc: "Stop sequences"],
      timeout: [type: :integer, default: 60_000, doc: "Request timeout in milliseconds"],
      stream: [type: :boolean, default: false, doc: "Enable streaming responses"],
      frequency_penalty: [type: :float, doc: "Frequency penalty parameter"],
      presence_penalty: [type: :float, doc: "Presence penalty parameter"],
      json_mode: [
        type: :boolean,
        default: false,
        doc: "Forces model to output valid JSON (provider-dependent)"
      ],
      verbose: [
        type: :boolean,
        default: false,
        doc: "Enable verbose logging"
      ],
      reasoning_effort: [
        type: {:in, [:low, :medium, :high]},
        doc: "Enable extended thinking with effort level (:low=1K, :medium=2K, :high=4K tokens)"
      ],
      thinking: [
        type: :map,
        doc: "Direct thinking config (e.g., %{type: \"enabled\", budget_tokens: 4096})"
      ]
    ]

  require Logger
  alias Jido.AI.Model
  alias Jido.AI.Prompt

  @impl true
  def on_before_validate_params(params) do
    with {:ok, model} <- validate_model(params.model),
         {:ok, prompt} <- Prompt.validate_prompt_opts(params.prompt) do
      {:ok, %{params | model: model, prompt: prompt}}
    else
      {:error, reason} ->
        Logger.error("ChatCompletion validation failed: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @impl true
  def run(params, _context) do
    # Validate required parameters exist
    with :ok <- validate_required_param(params, :model, "model"),
         :ok <- validate_required_param(params, :prompt, "prompt") do
      run_with_validated_params(params)
    else
      {:error, reason} -> {:error, reason}
    end
  end

  defp run_with_validated_params(params) do
    # Extract options from prompt if available
    prompt_opts =
      case params[:prompt] do
        %Prompt{options: options} when is_list(options) and length(options) > 0 ->
          Map.new(options)

        _ ->
          %{}
      end

    # Keep required parameters
    required_params = Map.take(params, [:model, :prompt, :tools])

    # Create a map with all optional parameters set to defaults
    # Priority: explicit params > prompt options > defaults
    params_with_defaults =
      %{
        temperature: 0.7,
        max_tokens: 1000,
        top_p: nil,
        stop: nil,
        timeout: 60_000,
        stream: false,
        max_retries: 0,
        frequency_penalty: nil,
        presence_penalty: nil,
        json_mode: false,
        verbose: false,
        reasoning_effort: nil,
        thinking: nil
      }
      # Apply prompt options over defaults
      |> Map.merge(prompt_opts)
      # Apply explicit params over prompt options
      |> Map.merge(
        Map.take(params, [
          :temperature,
          :max_tokens,
          :top_p,
          :stop,
          :timeout,
          :stream,
          :max_retries,
          :frequency_penalty,
          :presence_penalty,
          :json_mode,
          :verbose,
          :reasoning_effort,
          :thinking
        ])
      )
      # Always keep required params
      |> Map.merge(required_params)

    if params_with_defaults.verbose do
      Logger.info(
        "Running ReqLLM chat completion with params: #{inspect(params_with_defaults, pretty: true)}"
      )
    end

    with {:ok, model} <- validate_model(params_with_defaults.model),
         {:ok, messages} <- convert_messages(params_with_defaults.prompt),
         {:ok, req_options} <- build_req_llm_options(model, params_with_defaults),
         result <- call_reqllm(model, messages, req_options, params_with_defaults) do
      result
    else
      {:error, reason} ->
        Logger.error("Chat completion failed: #{inspect(reason)}")
        {:error, reason}
    end
  end

  # Private functions

  defp validate_required_param(params, key, name) do
    if Map.has_key?(params, key) do
      :ok
    else
      {:error, "Missing required parameter: #{name}"}
    end
  end

  defp validate_model(%LLMDB.Model{} = model), do: {:ok, model}
  defp validate_model(%Model{} = model), do: Model.from(model)
  defp validate_model(spec) when is_tuple(spec), do: Model.from(spec)

  defp validate_model(other) do
    Logger.error("Invalid model specification: #{inspect(other)}")
    {:error, "Invalid model specification: #{inspect(other)}"}
  end

  defp convert_messages(prompt) do
    messages =
      Prompt.render(prompt)
      |> Enum.map(fn msg ->
        %{role: msg.role, content: msg.content}
      end)

    {:ok, messages}
  end

  defp build_req_llm_options(_model, params) do
    # Build base options
    base_opts =
      []
      |> add_opt_if_present(:temperature, params.temperature)
      |> add_opt_if_present(:max_tokens, params.max_tokens)
      |> add_opt_if_present(:top_p, params.top_p)
      |> add_opt_if_present(:stop, params.stop)
      |> add_opt_if_present(:frequency_penalty, params.frequency_penalty)
      |> add_opt_if_present(:presence_penalty, params.presence_penalty)
      # Extended thinking options - ReqLLM handles translation
      |> add_opt_if_present(:reasoning_effort, params[:reasoning_effort])
      |> add_opt_if_present(:thinking, params[:thinking])

    # Add tools if provided
    opts_with_tools =
      case params[:tools] do
        tools when is_list(tools) and length(tools) > 0 ->
          # Convert Jido.Action modules to ReqLLM.Tool structs
          tool_specs =
            Enum.map(tools, fn tool ->
              # Create a ReqLLM.Tool struct from the Jido action module
              ReqLLM.Tool.new!(
                name: tool.name(),
                description: tool.description(),
                parameter_schema: tool.schema(),
                # Callback wraps the Jido action's run/2 function
                callback: fn args -> tool.run(args, %{}) end
              )
            end)

          Keyword.put(base_opts, :tools, tool_specs)

        _ ->
          base_opts
      end

    # ReqLLM handles authentication internally via environment variables
    {:ok, opts_with_tools}
  end

  defp add_opt_if_present(opts, _key, nil), do: opts
  defp add_opt_if_present(opts, key, value), do: Keyword.put(opts, key, value)

  defp call_reqllm(model, messages, req_options, params) do
    # Build model spec string from LLMDB.Model
    model_spec = "#{model.provider}:#{model.model}"

    if params.stream do
      call_streaming(model_spec, messages, req_options)
    else
      call_standard(model_spec, messages, req_options)
    end
  end

  defp call_standard(model_id, messages, req_options) do
    tools = Keyword.get(req_options, :tools, [])

    case ReqLLM.generate_text(model_id, messages, req_options) do
      {:ok, %ReqLLM.Response{finish_reason: :tool_calls} = response} when tools != [] ->
        # Model wants to call tools - use ReqLLM's built-in tool loop
        handle_tool_loop(response, tools, model_id, req_options)

      {:ok, response} ->
        # No tool calls or no tools - format response directly
        format_response(response)

      {:error, error} ->
        {:error, error}
    end
  end

  # Execute tools and continue conversation until we get a final text response
  defp handle_tool_loop(response, tools, model_id, req_options, depth \\ 0) do
    # Safety limit to prevent infinite loops
    if depth > 10 do
      {:error, "Tool loop exceeded maximum depth"}
    else
      # Get tool calls from response
      tool_calls = ReqLLM.Response.tool_calls(response)

      # Execute each tool and collect results as proper ReqLLM tool result messages
      # ReqLLM expects: %{role: :tool, tool_call_id: id, content: binary, name: tool_name}
      tool_result_messages =
        Enum.map(tool_calls, fn tool_call ->
          # ToolCall has nested function field: %{name: "...", arguments: "json string"}
          tool_name = tool_call.function.name
          # Arguments is a JSON string - parse it
          tool_args =
            case Jason.decode(tool_call.function.arguments || "{}") do
              {:ok, args} -> args
              {:error, _} -> %{}
            end

          # Find matching tool by name
          matching_tool = Enum.find(tools, fn t -> t.name == tool_name end)

          result =
            if matching_tool do
              case ReqLLM.Tool.execute(matching_tool, tool_args) do
                {:ok, result} -> Jason.encode!(result)
                {:error, reason} -> "Error: #{inspect(reason)}"
              end
            else
              "Error: Unknown tool #{tool_name}"
            end

          # Format as ReqLLM tool result message
          %{role: :tool, tool_call_id: tool_call.id, content: result, name: tool_name}
        end)

      # Continue conversation with tool results using response context
      # The context already includes the assistant message with tool_use
      updated_context = response.context

      # Append each tool result as a separate message (ReqLLM format)
      new_messages = ReqLLM.Context.to_list(updated_context) ++ tool_result_messages

      # Call again without tools in messages (context handles it)
      case ReqLLM.generate_text(model_id, new_messages, req_options) do
        {:ok, %ReqLLM.Response{finish_reason: :tool_calls} = new_response} ->
          # More tools requested - recurse
          handle_tool_loop(new_response, tools, model_id, req_options, depth + 1)

        {:ok, final_response} ->
          # Got final text response
          format_response(final_response)

        {:error, error} ->
          {:error, error}
      end
    end
  end

  defp call_streaming(model_id, messages, req_options) do
    opts_with_stream = Keyword.put(req_options, :stream, true)

    case ReqLLM.stream_text(model_id, messages, opts_with_stream) do
      {:ok, stream} ->
        # Return the stream wrapped in :ok tuple
        {:ok, stream}

      {:error, error} ->
        {:error, error}
    end
  end

  defp format_response(%{content: content, tool_calls: tool_calls}) when is_list(tool_calls) do
    formatted_tools =
      Enum.map(tool_calls, fn tool ->
        %{
          name: tool[:name] || tool["name"],
          arguments: tool[:arguments] || tool["arguments"],
          # Will be populated after execution
          result: nil
        }
      end)

    {:ok, %{content: content, tool_results: formatted_tools}}
  end

  defp format_response(%{content: content}) do
    {:ok, %{content: content, tool_results: []}}
  end

  # Handle ReqLLM.Response struct - extract content from message
  defp format_response(%ReqLLM.Response{} = response) do
    content = extract_content_from_response(response)
    thinking = ReqLLM.Response.thinking(response)
    {:ok, %{content: content, tool_results: [], thinking: thinking}}
  end

  defp format_response(response) when is_map(response) do
    # Fallback for other response formats
    content = response[:content] || response["content"] || ""
    {:ok, %{content: content, tool_results: []}}
  end

  # Extract text content from ReqLLM.Response message
  defp extract_content_from_response(%ReqLLM.Response{message: message}) when not is_nil(message) do
    case message do
      %{content: content} when is_binary(content) -> content
      %{content: blocks} when is_list(blocks) ->
        # Handle content blocks (text, thinking, tool_use, etc.)
        blocks
        |> Enum.filter(fn
          %{type: "text"} -> true
          %{type: :text} -> true
          _ -> false
        end)
        |> Enum.map(fn block -> Map.get(block, :text) || Map.get(block, "text") || "" end)
        |> Enum.join("\n")
      _ -> inspect(message)
    end
  end

  defp extract_content_from_response(_), do: ""
end
