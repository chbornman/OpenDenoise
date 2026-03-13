--[[
  OpenDenoise - AI denoising plugin for darktable

  Adds a module in lighttable with:
  - Mode selector (raw / bayer / post)
  - Strength slider (0.0 - 1.0)
  - Denoise button

  Selected images are processed via the OpenDenoise CLI, and the
  denoised results are imported back into the darktable library
  and grouped with the originals.

  Installation:
    1. Copy this file to ~/.config/darktable/lua/
    2. Add to ~/.config/darktable/luarc:
       require "opendenoise"
    OR use darktable's script manager to enable it.
]]

local dt = require "darktable"
local du = require "lib/dtutils"

-- ========================================================================
-- Configuration
-- ========================================================================

-- Path to the OpenDenoise CLI (adjust if needed)
local OPENDENOISE_CMD = "HIP_VISIBLE_DEVICES=0 python3 " ..
  os.getenv("HOME") .. "/projects/OpenDenoise/opendenoise/cli.py"

-- ========================================================================
-- Widgets
-- ========================================================================

local mode_selector = dt.new_widget("combobox") {
  label = "mode",
  tooltip = "raw: demosaic + denoise -> TIFF (best quality)\n" ..
            "bayer: denoise in Bayer space -> DNG (smallest, experimental)\n" ..
            "post: denoise exported TIFFs",
  "bayer", "raw", "post",
}

local strength_slider = dt.new_widget("slider") {
  label = "strength",
  tooltip = "Denoise strength (0 = no change, 1 = full denoise)",
  soft_min = 0.0,
  soft_max = 1.0,
  hard_min = 0.0,
  hard_max = 1.0,
  step = 0.05,
  digits = 2,
  value = 0.5,
}

local status_label = dt.new_widget("label") {
  label = "",
  selectable = false,
  halign = "start",
}

-- ========================================================================
-- Core logic
-- ========================================================================

local function denoise_images()
  local images = dt.gui.action_images

  if #images == 0 then
    dt.print("OpenDenoise: no images selected")
    return
  end

  local mode = mode_selector.value
  local strength = string.format("%.2f", strength_slider.value)
  local count = #images

  status_label.label = "Processing " .. count .. " image(s)..."
  dt.print("OpenDenoise: processing " .. count .. " image(s) in " .. mode .. " mode")

  -- Create a background job for progress
  local job = dt.gui.create_job("OpenDenoise", true)

  for i, image in ipairs(images) do
    job.percent = (i - 1) / count

    local input_path = image.path .. "/" .. image.filename
    -- Output goes next to the original
    local output_dir = image.path

    -- Build command
    local cmd = OPENDENOISE_CMD ..
      " " .. dt.util.shell_escape(input_path) ..
      " -o " .. dt.util.shell_escape(output_dir) ..
      " -m " .. mode ..
      " -s " .. strength ..
      " --no-suffix"

    dt.print_log("OpenDenoise: " .. cmd)

    local result = os.execute(cmd)

    if result then
      -- Determine output filename
      local basename = image.filename:match("(.+)%..+$")
      local out_ext = (mode == "bayer") and ".dng" or ".tif"
      local out_filename = basename .. out_ext
      local out_path = image.path .. "/" .. out_filename

      -- Check if file exists
      local f = io.open(out_path, "r")
      if f then
        f:close()
        -- Import into darktable
        local new_image = dt.database.import(out_path)
        if new_image then
          -- Group with original
          new_image:group_with(image)
          -- Tag it
          local tag = dt.tags.create("opendenoise")
          dt.tags.attach(tag, new_image)
          dt.print("OpenDenoise: imported " .. out_filename)
        end
      else
        dt.print_error("OpenDenoise: output not found: " .. out_path)
      end
    else
      dt.print_error("OpenDenoise: failed to process " .. image.filename)
    end
  end

  job.percent = 1.0
  job.valid = false
  status_label.label = "Done. " .. count .. " image(s) processed."
  dt.print("OpenDenoise: done. " .. count .. " image(s) processed.")
end

-- ========================================================================
-- Button
-- ========================================================================

local denoise_button = dt.new_widget("button") {
  label = "denoise",
  tooltip = "Run AI denoise on selected images",
  clicked_callback = function()
    denoise_images()
  end,
}

-- ========================================================================
-- Register module
-- ========================================================================

dt.register_lib(
  "opendenoise",              -- module name
  "OpenDenoise",              -- visible name
  true,                       -- expandable
  false,                      -- resetable
  {
    [dt.gui.views.lighttable] = {"DT_UI_CONTAINER_PANEL_RIGHT_CENTER", 600},
  },
  dt.new_widget("box") {
    orientation = "vertical",
    mode_selector,
    strength_slider,
    denoise_button,
    status_label,
  }
)

-- Register a shortcut
dt.register_event("opendenoise_shortcut", "shortcut",
  function()
    denoise_images()
  end,
  "OpenDenoise selected images"
)

dt.print_log("OpenDenoise plugin loaded")
