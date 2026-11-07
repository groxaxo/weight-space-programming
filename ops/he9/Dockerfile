FROM python:3.11.17-slim@sha256:b7a9e5f6a93d6d4a31f129bbd4bdee69efccb6ded0fb230d78aafaf37af98377
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git ripgrep libstdc++6 && rm -rf /var/lib/apt/lists/*
# Supply only the authenticated ELF, without a personal launcher or profiles.
COPY opencode /usr/local/bin/opencode
RUN chmod 0755 /usr/local/bin/opencode && test "$(sha256sum /usr/local/bin/opencode | cut -d' ' -f1)" = 538fcb1b6408c8d4608822acee19f7ff5b3a3eb3383c0633c4f06170857312a7 && test "$(opencode --version)" = 1.19.4 && python3 --version
ENV HOME=/workspace OPENCODE_DISABLE_CLAUDE_CODE=true OPENCODE_DISABLE_EXTERNAL_SKILLS=true OPENCODE_DISABLE_AUTOUPDATE=true OPENCODE_DISABLE_MODELS_FETCH=true OPENCODE_DISABLE_SHARE=true OPENCODE_DISABLE_DEFAULT_PLUGINS=true
WORKDIR /work
ENTRYPOINT ["/usr/local/bin/opencode"]
